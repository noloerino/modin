"""
Module contains ``OpportunisticQueryCompiler`` class.

Unlike the ``PandasQueryCompiler``, where queries return concrete DataFrame
objects, this class instead produces an object containing a query _plan_, which
must be executed in order to produce a DataFrame.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Optional
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


@dataclass
class DfOp:
    def apply(self, df):
        """Apply this operation to the dataframe in question."""
        pass


# Assume for now that all dataframes are immutable, so we can just maintain a global store of histograms
# In the future, we can store a histogram dataframe as a member of each dataframe, then invalidate it when
# an in-place updating function is invoked
# TODO do something for non-integer types?
histograms = {} # Maps id(df): histogram

def histogram(df, bins=10): # Technically, should key the dict on #bins too, but whatever
    """
    Computes and caches a dataframe containing bin information for each numeric column in df.
    Every column is treated as its own dataframe.
    
    The count of null/NaN fields in each column is also stored.
    """
    for colname, tpe in df.dtypes.iteritems():
        if "float" in str(tpe) or "int" in str(tpe):
            hist_df = pd.DataFrame(index=["none", "size"
                                 ] + [f"bin{i}" for i in range(bins)] + [f"__lbound_{i}" for i in range(bins + 1)])
            col = df[colname]
            notna_mask = col.notna()
            nacol = pd.Series({"none": col.isna().sum()})
            if notna_mask.any():
                vals, bounds = pd.cut(col[notna_mask].values, bins=bins, retbins=True, labels=[f"bin{i}" for i in range(bins)])
                binned = pd.Series(vals.value_counts()).append(pd.Series({f"__lbound_{i}": b for i, b in enumerate(bounds)}))
            else:
                binned = pd.Series({f"bin{i}": 0 for i in range(bins)}).append(
                    pd.Series({f"__lbound_{i}": 0 for i in range(bins + 1)}))
            # Need to call pd.Series again in order to wrap the vanilla pandas frame
            nacol = nacol.append(binned).append(pd.Series({"size": df.size}))
            # ignore the bound for the "none" column
            hist_df = hist_df.join(pd.DataFrame({"counts": nacol}), how="left")
            histograms[id(col)] = hist_df


class Comparison(Enum):
    EQ = 0
    NE = 1
    LT = 2
    LTE = 3
    GT = 4
    GTE = 5
    
    @staticmethod
    def size_estimate(df, comp, value):
        if value is None and comp == Comparison.NE:
            hist = histograms[id(df)]
            size = hist["size"]
            na_size = hist["none"]
            return 1 - (na_size / size)
        else:
            raise NotImplementedError()
    
    @staticmethod
    def get_mask(df, comp, value):
        if value is None:
            if comp == Comparison.EQ:
                return Map.register(pd.DataFrame.isna, dtypes=np.bool)
            elif comp == Comparison.NE:
                return Map.register(pd.DataFrame.notna, dtypes=np.bool)
            else:
                raise NotImplementedError()
        elif comp == Comparison.EQ:
            return df == value
        elif comp == Comparison.NE:
            return df != value
        elif comp == Comparison.LT:
            return df < value
        elif comp == Comparison.LTE:
            return df <= value
        elif comp == Comparison.GT:
            return df > value
        elif comp == Comparison.GTE:
            return df >= value


# Produces a dataframe where elements are T/F according to some comparison
# Probably poorly named
@dataclass
class MaskOp(DfOp):
    comp: Comparison
    value: Optional[int]

    def __init__(self, comp, value):
        self.comp = comp
        self.value = value
    
    def apply(self, df):
        print("*mask")
        return Comparison.get_mask(df, self.comp, self.value)


# Filters elements of a dataframe based on some mask described by a QueryPlan
# Probably poorly named
@dataclass
class FilterOp(DfOp):
    """The other QueryPlan will produce this mask"""
    mask: "QueryPlan"

    def __init__(self, mask):
        self.mask = mask

    def apply(self, df):
        print("*filter")
        def where_builder_series(df, cond):
            return df.where(cond, other)
        return df.binary_op(where_builder_series, self.mask.execute(), join_type="left")


@dataclass
class SelectCol(DfOp):
    colname: str

    def __init__(self, colname):
        self.colname = colname

    def apply(self, df):
        print("*select")
        return df.mask(col_indices=[self.colname])

        
@dataclass
class QueryPlan:
    # TODO make this a dag or tree or something instead
    ops: Tuple[DfOp]

    def __init__(self):
        self.ops = ()

    def pretty_str(self, idlevel=0):
        def indents():
            return "  " * idlevel
        s = indents() + "QueryPlan(df_id=" + str(id(self.df))
        if self.ops:
            s += "\n"
        idlevel += 1
        for op in self.ops:
            tpe = type(op)
            if tpe is FilterOp:
                s += indents() + "Filter(\n"
                s += op.mask.pretty_str(idlevel=idlevel+1) + "\n"
                s += indents() + "),\n" 
            else:
                s += indents() + str(op) + ",\n"
        idlevel -= 1
        s += indents() + ")"
        return s

    def append_op(self, new_df, op):
        newplan = QueryPlan(new_df)
        newplan.ops = self.ops + (op,)
        return newplan

    def copy(self, new_df):
        newplan = QueryPlan(new_df)
        newplan.ops = self.ops
        return newplan

    def execute(self):
        df = self.df
        for op in self.ops:
            df = op.apply(df)
        return df


class OpportunisticPandasQueryCompiler(PandasQueryCompiler):
    """
    Opportunistic query compiler for the pandas storage format.gg

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

    verbose = True

    def __str__(self):
        return f"OpportunisticPandasQueryCompiler({self._plan})"

    def __repr__(self):
        return str(self)

    def print_plan(self):
        print(self._plan)

    def _print_op_start(self, msg):
        if self.verbose:
            print(f"!!{msg} on qc")

    def _info(self, *args):
        if self.verbose:
            print("!!!", *args)

    def getitem_array(self, key):
        self._print_op_start("getitem_array")
        self._info("key is", key)
        raise Exception()

    def getitem_column_array(self, key, numeric=False):
        self._print_op_start("getitem_column_array")
        if numeric:
            raise NotImplementedError()
        else:
            self._info("getitem with key:", key)
            assert len(key) == 1 and type(key[0]) == str, "only colnames allowed for now"
            return OpportunisticPandasQueryCompiler(
                self._modin_frame,
                lambda df: self._plan.append_op(df, SelectCol(key[0]))
            )

    def notna(self):
        self._print_op_start("notna")
        return OpportunisticPandasQueryCompiler(
            self._modin_frame,
            lambda df: self._plan.append_op(df, MaskOp(Comparison.NE, None))
        )

    def where(self, cond, other, **kwargs):
        assert isinstance(
            cond, type(self)
        ), "Must have the same QueryCompiler subclass to perform this operation"
        # "else" branch of PandasQueryCompiler.where
        assert isinstance(other, pd.Series), "other must be Series"
        return OpportunisticPandasQueryCompiler(
            self._modin_frame,
            lambda df: self._plan.append_op(df, FilterOp(cond._plan))
        )
