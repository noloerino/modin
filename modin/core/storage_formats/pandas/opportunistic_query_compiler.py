"""
Module contains ``OpportunisticQueryCompiler`` class.

Unlike the ``PandasQueryCompiler``, where queries return concrete DataFrame
objects, this class instead produces an object containing a query _plan_, which
must be executed in order to produce a DataFrame.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd

from modin.core.dataframe.algebra import (
    # Fold,
    Map,
    # MapReduce,
    # Reduction,
    Binary,
    # GroupByReduce,
    # groupby_reduce_functions,
)
from .query_compiler import PandasQueryCompiler


@dataclass(frozen=True)
class DfOp:
    def apply(self, df):
        """Apply this operation to the dataframe in question."""
        pass

    def other(self):
        return None

    _verbose = False

    def _info(self, msg, *args):
        if self._verbose:
            print(msg, *args)


# Assume for now that all dataframes are immutable, so we can just maintain a global store of histograms
# In the future, we can store a histogram dataframe as a member of each dataframe, then invalidate it when
# an in-place updating function is invoked
# TODO do something for non-integer types?
_histograms = {} # Maps hash((id(df), colname)): histogram

def hist_key(df, colname):
    return ((id(df), colname))

def histogram(df, maxbins=10):
    """
    Computes and caches a dataframe containing bin information for each numeric column in df.
    Every column is treated as its own dataframe.
    
    The count of null/NaN fields in each column is also stored.

    Fields in resulting series:
    - bin_count: number of bins
    - none: number of None/NaN values
    - size: the number of items in the dataframe
    - count_{i}: number of values in ith bin
    - distinct_{i}: the number of distinct values in ith bin
    - lbound_{i}: left (lower) bound for ith bin
                  lbound_0 is the minimum
                  lbound_{bin_count} is the maximum

    TODO combine series for cols of the same df into a single dataframe
    """
    for colname, tpe in df.dtypes.iteritems():
        mdf = df._query_compiler._modin_frame
        hk = hist_key(mdf, colname)
        if hk in _histograms:
            continue
        if "float" in str(tpe) or "int" in str(tpe):
            col = df._getitem_column(colname)._query_compiler._plan.execute().squeeze(axis=1)
            bins = min(maxbins, col.size)
            bin_labels = [f"count_{i}" for i in range(bins)]
            distinct_labels = [f"distinct_{i}" for i in range(bins)]
            bound_labels = [f"lbound_{i}" for i in range(bins + 1)]
            hist_df = pd.DataFrame(index=["none", "items", "bin_count"] + bin_labels + distinct_labels + bound_labels)
            notna_mask = col.notna()
            nacol = pd.Series({"bin_count": bins, "none": col.isna().sum(), "items": col.size})
            if notna_mask.any():
                vals, bounds = pd.cut(col[notna_mask].values, bins=bins, retbins=True, labels=bin_labels, include_lowest=True)
                # TODO find some way to avoid doing a second pass for unique
                distinct = pd.cut(pd.unique(col[notna_mask].values), bins=bounds, labels=distinct_labels)
                binned = pd.Series(vals.value_counts()).append(
                    pd.Series(distinct.value_counts()).append(
                    pd.Series({l: b for l, b in zip(bound_labels, bounds)})))
            else:
                binned = pd.Series({l: 0 for l in bin_labels + distinct_labels + bound_labels})
            nacol = nacol.append(binned)
            # ignore the bound for the "none" column
            hist_df = hist_df.join(pd.DataFrame({"counts": nacol}), how="left")
            _histograms[hk] = hist_df


def get_histogram(df, colname):
    return _histograms[hist_key(df, colname)]


_stats_queue = []

class _StatsManager:
    def has_next(self):
        return len(_stats_queue) > 0

    def size(self):
        return len(_stats_queue)

    def compute_next(self):
        if _stats_queue:
            histogram(_stats_queue.pop(0))

    def compute_all(self):
        """
        Attempts to run background statistics collection on the next dataframe
        in _stats_queue.
        """
        while _stats_queue:
            self.compute_next()
        # print(_stats_queue)
        # print(_histograms)

    def get_all(self):
        return _histograms

    def clear_all(self):
        global _histograms
        global _stats_queue
        del _histograms
        del _stats_queue
        _histograms = {}
        _stats_queue = []

stats_manager = _StatsManager() # singleton, exposed to API

def _queue_df(df):
    # only queue the dataframe if it was constructed by the user (not the
    # query compiler as an intermediate computation)
    # this means the plan must be empty
    if len(df._query_compiler._plan.ops) == 0:
        _stats_queue.append(df)


class Comparison(Enum):
    EQ = 0
    NE = 1
    LT = 2
    LE = 3
    GT = 4
    GE = 5

    def size_estimate(self, df, colname, value):
        """
        Returns a cardinality estimate for this comparison on a particular
        column of the provided dataframe.

        Raises KeyError if the histogram has not yet been generated.
        TODO return from uniform distribution instead?

        Raises NotImplementedError if, well, you know, estimation isn't implemented
        for the operation.
        """
        hist = _histograms[hist_key(df, colname)]["counts"]
        comp = self
        if value is None:
            size = hist["items"]
            na_size = hist["none"]
            if comp == Comparison.NE:
                return size - na_size
            elif comp == Comparison.EQ:
                return na_size
            else:
                raise NotImplementedError()
        bins = int(hist["bin_count"])
        if bins == 0:
            return
        items = hist["items"] - hist["none"]
        maxval = hist[f"lbound_{bins}"]
        minval = hist[f"lbound_0"]
        binid = -1
        # probaly should binary search instead, but it's just 10 elements
        for i in range(bins):
            lb = hist[f"lbound_{i}"]
            ub = hist[f"lbound_{i + 1}"]
            # we use <= rather than < because we specified include_lowest in pd.cut
            if lb <= value <= ub:
                binid = i
                break
        # microoptimization: make these computations lazy
        # these estimates are as described in 13.3 of abraham/silberschatz, p592-594
        eq_estimate = hist[f"count_{binid}"] / hist[f"distinct_{binid}"] if binid != -1 else 0
        le_estimate = 0
        if value > maxval:
            eq_estimate = 0
            le_estimate = items
        if value < minval:
            eq_estimate = 0
            le_estimate = 0
        if binid != -1:
            # iterate up to our current bin
            for i in range(binid):
                le_estimate += hist[f"count_{i}"]
            # within our current bin, assume a uniform distribution
            lb = hist[f"lbound_{binid}"]
            ub = hist[f"lbound_{binid + 1}"]
            le_estimate += (value - lb) / (ub - lb)
        eq_estimate = max(0, min(items, eq_estimate))
        le_estimate = max(0, min(items, le_estimate))
        if comp == Comparison.EQ:
            if value > maxval or value < minval:
                return 0
            return eq_estimate
        elif comp == Comparison.NE:
            return items - eq_estimate
        elif comp == Comparison.LT:
            if value >= maxval:
                return items
            return le_estimate - eq_estimate
        elif comp == Comparison.LE:
            if value > maxval:
                return items
            return le_estimate
        elif comp == Comparison.GT:
            if value <= minval:
                return items
            return items - le_estimate
        elif comp == Comparison.GE:
            if value < minval:
                return items
            return items - le_estimate + eq_estimate
        raise NotImplementedError()
    
    def get_mask(self, df, value):
        comp = self
        if value is None:
            if comp == Comparison.EQ:
                return df.map(lambda x: pd.DataFrame.isna(x, dtypes=np.bool))
                # Map.register(pd.DataFrame.isna, dtypes=np.bool)
            elif comp == Comparison.NE:
                return df.map(lambda x: pd.DataFrame.notna(x))
                # return Map.register(pd.DataFrame.notna, dtypes=np.bool)
            else:
                raise NotImplementedError()
        # Assume "value" is numeric, rather than another dataframe
        elif comp == Comparison.EQ:
            return df.map(lambda x: pd.DataFrame.eq(x, value))
        elif comp == Comparison.NE:
            return df.map(lambda x: pd.DataFrame.ne(x, value))
        elif comp == Comparison.LT:
            return df.map(lambda x: pd.DataFrame.lt(x, value))
        elif comp == Comparison.LE:
            return df.map(lambda x: pd.DataFrame.le(x, value))
        elif comp == Comparison.GT:
            return df.map(lambda x: pd.DataFrame.gt(x, value))
        elif comp == Comparison.GE:
            return df.map(lambda x: pd.DataFrame.ge(x, value))
        raise NotImplementedError()


# Produces a dataframe where elements are T/F according to some comparison
@dataclass(frozen=True)
class CompOp(DfOp):
    comp: Comparison
    value: Optional[int]
    _hint_colname: Optional[str]
    _kwargs: Dict[str, object] = field(repr=False) # unused for now
    
    def apply(self, df):
        self._info(f"*mask on df {id(df)}")
        return self.comp.get_mask(df, self.value)

    def size_estimate(self, df):
        if self._hint_colname:
            try:
                return self.comp.size_estimate(df, self._hint_colname, self.value)
            except KeyError:
                print("MISSING HISTOGRAM:", id(df), self._hint_colname)
                pass
        return len(df.index)

    def __hash__(self):
        return hash((self.comp, self.value))


# Filters elements of a dataframe based on some mask described by a QueryPlan
# Probably poorly named
@dataclass(frozen=True)
class FilterOp(DfOp):
    """
    Because the df.binary_op builder function expects another query compiler object, we need
    to save the whole thing rather than just the plan
    """
    cond: "OpportunisticPandasQueryCompiler"
    """The other QueryPlan will produce this mask"""
    # mask: "QueryPlan"

    def __init__(self, cond, other, **kwargs):
        """
        From pandas docs:
            Entries where cond is False are replaced with corresponding value from other. If other
            is callable, it is computed on the Series/DataFrame and should return scalar or
            Series/DataFrame. The callable must not change input Series/DataFrame (though pandas
            doesn’t check it).

        i have no idea what `other` here is though
        """
        self.cond = cond
        self.other = other
        self.kwargs = kwargs

    def other(self):
        return self.cond

    def apply(self, df):
        self._info(f"*filter on df {id(df)}")
        def where_builder_series(df, cond):
            return df.where(cond, self.other, **self.kwargs)
        cond_df = self.cond._plan._execute()
        return df.binary_op(where_builder_series, cond_df, join_type="left")


@dataclass(frozen=True)
class SelectCol(DfOp):
    colname: str

    def apply(self, df):
        self._info(f"*select on df {id(df)}")
        return df.mask(col_indices=[self.colname])


# Masks a dataframe according to the provided row/column indices
@dataclass(frozen=True)
class MaskOp(DfOp):
    # idk the types
    # row_numeric_idx: 
    # col_numeric_idx:
    """
    (hopefully) temporary hack: this QC produces a DF that's a 1-d bool array, and
    we select every row where this DF is true and drop rows that are false
    """
    rows: "OpportunisticPandasQueryCompiler"

    def apply(self, df):
        # similar to PandasQueryCompiler.getitem_array, but hopefully more efficient
        # by converting to np array instead of a new dataframe
        if df.index.empty:
            return df
        mask_rows = self.rows._plan._execute() #.squeeze(axis=1)
        idx = mask_rows.index
        if idx.empty or len(idx) == 0:
            key = idx
        else:
            mask_rows = mask_rows.transpose().to_numpy()
            assert len(mask_rows) == 1
            key = pd.RangeIndex(len(df.index))[mask_rows[0]]
        return df.mask(row_numeric_idx=key)

    def other(self):
        return self.rows

@dataclass(frozen=True)
class InnerJoin(DfOp):
    """
    Performs an inner join on two dataframes.
    """
    right: "OpportunisticPandasQueryCompiler"

    def apply(self, df):
        right = self.right

        def map_func(left, right=right._plan.execute(), kwargs={}):
            return pd.merge(left, right, **kwargs)
        new_self = OpportunisticPandasQueryCompiler(
            df.apply_full_axis(1, map_func)
        )
        return new_self.reset_index(drop=True)._plan._execute()

    def other(self):
        return self.right

@dataclass(frozen=True)
class MeanOp(DfOp):
    """
    Calculates the mean across a particular axis of the dataframe.
    """
    axis: int

    def apply(self, df):
        """copied from PandasQueryCompiler"""
        pandas = pd
        axis = self.axis
        kwargs = {}
        if kwargs.get("level") is not None:
            return self.default_to_pandas(pandas.DataFrame.mean, axis=axis, **kwargs)
        skipna = kwargs.get("skipna", True)
        # TODO-FIX: this function may work incorrectly with user-defined "numeric" values.
        # Since `count(numeric_only=True)` discards all unknown "numeric" types, we can get incorrect
        # divisor inside the reduce function.
        def map_fn(df, **kwargs):
            """
            Perform Map phase of the `mean`.

            Compute sum and number of elements in a given partition.
            """
            result = pandas.DataFrame(
                {
                    "sum": df.sum(axis=axis, skipna=skipna),
                    "count": df.count(axis=axis, numeric_only=True),
                }
            )
            return result if axis else result.T

        def reduce_fn(df, **kwargs):
            """
            Perform Reduce phase of the `mean`.

            Compute sum for all the the partitions and divide it to
            the total number of elements.
            """
            sum_cols = df["sum"] if axis else df.loc["sum"]
            count_cols = df["count"] if axis else df.loc["count"]

            if not isinstance(sum_cols, pandas.Series):
                # If we got `NaN` as the result of the sum in any axis partition,
                # then we must consider the whole sum as `NaN`, so setting `skipna=False`
                sum_cols = sum_cols.sum(axis=axis, skipna=False)
                count_cols = count_cols.sum(axis=axis, skipna=False)
            return sum_cols / count_cols
        # monkeypatch for MapReduce.register + df._reduce_dimension
        # doesn't really work
        data = df.map_reduce(
            axis,
            lambda x: map_fn(x, **kwargs),
            lambda y: reduce_fn(y, **kwargs),
        ).transpose()
        return data

        
# maps QueryPlan hash to executed result
# this cannot be stored as just a field on the class because we might construct
# the same queryplan from 2 different instances
# values are pairs of (result_df, result_pandas_df), where pandas_df may be none
_plans = {}

# map of QueryPlan -> (QueryPlan, int), where second int is cost
_bestplans = {}

@dataclass
class QueryPlan:
    # TODO make this a dag or tree or something instead
    ops: Tuple[DfOp]

    def __init__(self, df, ops=()):
        # tuples are immutable, so it's ok if the kwargs is duplicated
        self.ops = ops
        self.df = df

    def __hash__(self):
        return hash((self.ops, id(self.df)))

    def pretty_str(self, idlevel=0):
        def indents():
            return "  " * idlevel
        s = indents() + "QueryPlan(df_id=" + str(id(self.df))
        if self.ops:
            s += "\n"
        idlevel += 1
        for op in self.ops:
            other = op.other()
            if other:
                s += indents() + type(op).__name__ + "(\n"
                s += other._plan.pretty_str(idlevel=idlevel+1) + "\n"
                s += indents() + "),\n" 
            else:
                s += indents() + str(op) + ",\n"
        idlevel -= 1
        if self.ops:
            s += indents()
        s += ")"
        return s

    def append_op(self, new_df, op):
        return QueryPlan(new_df, self.ops + (op,))

    def copy(self, new_df):
        return QueryPlan(new_df, self.ops)

    def _card_estimates(self):
        # Computes (row, col) cardinality estimates
        row_card = len(self.df.index)
        # print("BEGIN counting cost for qp", id(self))
        for o in self.ops:
            if isinstance(o, CompOp):
                # Even though technically the cardinality doesn't change, we
                # can figure out how many rows would be true in this result
                row_card += o.size_estimate(self.df)
            elif isinstance(o, SelectCol):
                pass
                # print("selectcol estimate 1", o)
            elif isinstance(o, FilterOp):
                o_rc = o.cond._plan._card_estimates()
                row_card += o_rc
                # print("filter estimates:", o, row_card)
            elif isinstance(o, MaskOp):
                o_rc = o.rows._plan._card_estimates()
                row_card += o_rc
                # print("mask estimate:", o, row_card)
            elif isinstance(o, InnerJoin):
                # TODO column count should go up generally, for now we just penalize
                # the row cardinality heuristic for simplicity
                row_card *= 2
            else:
                raise NotImplementedError(f"could not compute cardinality estimate for {o}")
        # print("END counting cost for qp", id(self), row_card)
        return row_card

    def optimize(self):
        """
        Greedy algorithm that reorders query plan operations according to equivalence rules.

        TODO do something about metadata tracking? e.g. pushing column selection across a mask
        """
        if self in _bestplans:
            return _bestplans[self]
        all_ops = list(self.ops)
        # (hacky) hardcoded transformation for sequential mask operations
        for i in range(len(all_ops) - 1):
            # Attempts to perform optimizations for this pattern:
            # MaskOp1(Select1, Comp1)
            # MaskOp2(MaskOp1, Select2, Comp2)
            # which can be transformed into
            # MaskOp2(Select2, Comp2)
            # MaskOp1(MaskOp2, Select1, Comp1)
            # fixing this requires a more robust query plan expression tree
            o1, o2 = all_ops[i:i + 2]
            if isinstance(o1, MaskOp) and isinstance(o2, MaskOp):
                try:
                    sel1, comp1 = o1.other()._plan.ops
                    maybe_mo1, sel2, comp2 = o2.other()._plan.ops
                    if maybe_mo1 != o1:
                        continue
                    cost1 = QueryPlan(o1.other()._modin_frame, (o1, o2))._card_estimates()

                    o2c = o2.other().copy()
                    o2df = o2c._modin_frame
                    o2c._plan = QueryPlan(o2df, (sel2, comp2))
                    o2p = MaskOp(o2c)
                    o1c = o1.other().copy()
                    o1c._plan = QueryPlan(o1c._modin_frame, (o2p, sel1, comp1))
                    o1p = MaskOp(o1c)
                    # print("candidate swapped plan:")
                    candidate = QueryPlan(o2df, (o2p, o1p))
                    # print(candidate.pretty_str())
                    cost2 = candidate._card_estimates()
                    # print("cost1", cost1)
                    # print("cost2", cost2)
                    if cost2 < cost1:
                        all_ops[i] = o2p
                        all_ops[i + 1] = o1p
                except ValueError:
                    # destructuring failed, whatever
                    continue

        # Next, recursively optimize
        for i, op in enumerate(all_ops):
            other_qc = op.other()
            if not other_qc:
                continue
            new_plan = other_qc._plan.optimize()
            new_qc = OpportunisticPandasQueryCompiler(
                other_qc._modin_frame,
                new_plan_factory=lambda _df: new_plan
            )
            if isinstance(op, InnerJoin):
                all_ops[i] = InnerJoin(new_qc)
            elif isinstance(op, MaskOp):
                all_ops[i] = MaskOp(new_qc)
            elif isinstance(op, FilterOp):
                all_ops[i] = FilterOp(new_qc, other_qc.other, **other_qc.kwargs)
            else:
                raise NotImplementedError()

        new_plan = QueryPlan(self.df, tuple(all_ops))
        _bestplans[self] = new_plan
        _bestplans[new_plan] = new_plan
        return new_plan

        """
        Returns a new QueryPlan object that is optimized based on the dynamic programming algorithm
        given in figure 13.7 of Abraham/Silberschatz.
        """
        """
        procedure FindBestPlan(S)
            if (bestplan[S].cost != infinity) /* bestplan[S] already computed */
                return bestplan[S]
            if (S contains only 1 relation)
                set bestplan[S].plan and bestplan[S].cost based on best way of accessing S
            else for each non-empty subset S1 of S such that S1 != S
                P1 = FindBestPlan(S1)
                P2 = FindBestPlan(S - S1)
                A = best algorithm for joining results of P1 and P2
                cost = P1.cost + P2.cost + cost o fA
                if cost < bestplan[S].cost
                    bestplan[S].cost = cost
                    bestplan[S].plan = "execute P1.plan; execute P2.plan; join results of P1 and
                                        P2 using A"
            return bestplan[S]
        """

    def _execute(self):
        """
        Returns a _modin_frame object (should be of type PandasOnRayDataFrame).
        Memoizes the result.

        If being presented to the user, use the public execute() instead, which
        will attempt to check if the to_pandas conversion was cached.
        """
        if self in _plans:
            # print("cached non-pd")
            return _plans[self][0]
        df = self.df
        i = 0
        for op in self.ops:
            # print(op)
            # old_id = id(df)
            df = op.apply(df)
            # self._info(f"***iteration {i} w/ {op}: {old_id} -> {id(df)}")
            # self._info("***new df:")
            # self._info(df.to_pandas().head())
            # self._info()
            i += 1
        _plans[self] = (df, None)
        return df

    def execute(self, clean=False):
        if self not in _plans or clean:
            # memoizes _plans[self][0]
            self._execute()
        if _plans[self][1] is not None and not clean:
            # print("using cached pd")
            return _plans[self][1]
        result_df = _plans[self][0]
        if isinstance(result_df, pd.Series) or isinstance(result_df, pd.DataFrame):
            df_pandas = result_df
        else:
            df_pandas = result_df.to_pandas()
        _plans[self] = (result_df, df_pandas)
        return df_pandas

class OpportunisticPandasQueryCompiler(PandasQueryCompiler):
    """
    Opportunistic query compiler for the pandas storage format.

    Whereas ``PandasQueryCompiler`` translates common query compiler API calls
    directly into DataFrame Algebra operations, this instead returns a query plan.

    Parameters
    ----------
    modin_frame : PandasDataFrame
        Modin Frame for which to produce a query plan

    new_plan_factory : PandasDataFrame -> QueryPlan
        Function that returns the query plan to apply to modin_frame. This is a
        function rather than a concrete object because default parameters share
        the same object.
    """
    def __init__(self, modin_frame, new_plan_factory=lambda df: QueryPlan(df)):
        super().__init__(modin_frame)
        self._plan = new_plan_factory(modin_frame)
        self._print_op_start("new df")
        self._info("with plan", self._plan, ", id=", id(self))

    def copy(self):
        # Because the compiler plan is lazy, we're fine with copying the frame
        # (this may change later)
        return self.__constructor__(self._modin_frame, new_plan_factory=self._plan.copy)

    lazy_execution = True

    _verbose = False

    def __str__(self):
        return f"OpportunisticPandasQueryCompiler({self._plan})"

    def __repr__(self):
        return str(self)

    def print_plan(self):
        print(self._plan)

    def _new_qc_with_op(self, op):
        return OpportunisticPandasQueryCompiler(
            self._modin_frame,
            lambda df: self._plan.append_op(df, op)
        )

    def _comp_with_other(self, comp, other, kwargs):
        return self._new_qc_with_op(CompOp(comp, other, self._colname_hint(), kwargs))

    def _print_op_start(self, msg):
        if self._verbose:
            print(f"!! {msg} on qc")

    def _info(self, *args):
        if self._verbose:
            print("!!!", *args)

    def getitem_array(self, key):
        self._print_op_start("getitem_array")
        self._info("key is", key)
        # assume this is a bool_indexer, which means rows get dropped
        # ignore checks for now :/
        assert isinstance(key, type(self)), "getitem_array key must also be the same QueryCompiler type"
        return self.getitem_row_array(key)

    def getitem_column_array(self, key, numeric=False):
        self._print_op_start("getitem_column_array")
        if numeric:
            raise NotImplementedError()
        else:
            self._info("getitem with key:", key)
            assert len(key) == 1 and type(key[0]) == str, "only colnames allowed for now"
            return self._new_qc_with_op(SelectCol(key[0]))

    def getitem_row_array(self, key):
        self._print_op_start("getitem_row_array")
        self._info("key is", key)
        assert isinstance(key, type(self)), "getitem_row_array key must also be the same QueryCompiler"
        return self._new_qc_with_op(MaskOp(key))

    def _colname_hint(self):
        ops = self._plan.ops
        # If the last operation was a select, then if the next operation is
        # a comparison, we can use the column name of the select for histograms
        if ops and isinstance(ops[-1], SelectCol):
            return ops[-1].colname
        return None

    def notna(self):
        self._print_op_start("notna")
        return self._comp_with_other(Comparison.NE, None, {})

    def lt(self, other, **kwargs):
        self._print_op_start("lt")
        return self._comp_with_other(Comparison.LT, other, kwargs)

    def le(self, other, **kwargs):
        self._print_op_start("le")
        return self._comp_with_other(Comparison.LE, other, kwargs)

    def gt(self, other, **kwargs):
        self._print_op_start("gt")
        return self._comp_with_other(Comparison.GT, other, kwargs)

    def ge(self, other, **kwargs):
        self._print_op_start("ge")
        return self._comp_with_other(Comparison.GE, other, kwargs)

    def eq(self, other, **kwargs):
        self._print_op_start("eq")
        return self._comp_with_other(Comparison.EQ, other, kwargs)

    def ne(self, other, **kwargs):
        self._print_op_start("ne")
        return self._comp_with_other(Comparison.NE, other, kwargs)

    def where(self, cond, other, **kwargs):
        self._print_op_start("where")
        assert isinstance(
            cond, type(self)
        ), "Must have the same QueryCompiler subclass to perform this operation"
        # "else" branch of PandasQueryCompiler.where
        assert isinstance(other, pd.Series), "other must be Series"
        return self._new_qc_with_op(FilterOp(cond, other, **kwargs))

    def merge(self, other, how="inner", **kwargs):
        assert how == "inner", "only inner joins are currently supported"
        assert isinstance(other, type(self)), "merge must be with other df with same qc type"
        return self._new_qc_with_op(InnerJoin(other))

    def mean(self, axis, **kwargs):
        self._print_op_start("mean")
        if axis is None:
            axis = kwargs.get("axis", 0)
        return self._new_qc_with_op(MeanOp(axis))
