"""Methods for interacting with mysql db_emorep.

DbConnect : connect to and interact with db_emorep on mysql server
MysqlUpdate : update db_emorep tables

"""
# %%
import os
import pandas as pd
from typing import Type
import mysql.connector
from contextlib import contextmanager


# %%
class DbConnect:
    """Connect to mysql server and update db_emorep.

    Methods
    -------
    exec_many()
        Update mysql db_emorep.tbl_* with multiple values

    Notes
    -----
    Requires global var 'SQL_PASS' to contain user password
    for mysql db_emorep.

    Example
    -------
    db_con = sql_database.DbConnect()
    sql_cmd = (
        "insert ignore into ref_subj (subj_id, subj_name) values (%s, %s)"
    )
    tbl_input = [(9, "ER0009"), (16, "ER0016")]
    db_con.exec_many(sql_cmd, tbl_input)

    """

    def __init__(self):
        """Set db_con attr as mysql connection."""
        try:
            os.environ["SQL_PASS"]
        except KeyError as e:
            raise Exception(
                "No global variable 'SQL_PASS' defined in user env"
            ) from e

        self.con = mysql.connector.connect(
            host="localhost",
            user=os.environ["USER"],
            password=os.environ["SQL_PASS"],
            database="db_emorep",
        )

    @contextmanager
    def connect(self):
        """Yield cursor."""
        cursor = self.con.cursor()
        try:
            yield cursor
        finally:
            cursor.close()

    def exec_many(self, sql_cmd: str, value_list: list):
        """Update db_emorep via executemany."""
        with self.connect() as cur:
            cur.executemany(sql_cmd, value_list)
            self.con.commit()

    def fetch_all(self, sql_cmd: str, col_names: list) -> pd.DataFrame:
        """Return dataframe from query output."""
        with self.connect() as cur:
            cur.execute(sql_cmd)
            df = pd.DataFrame(cur.fetchall(), columns=col_names)
        return df

    def close_con(self):
        """Title."""
        self.con.close()


# %%
class _RefMaps:
    """Supply mappings to SQL reference table values."""

    def __init__(self, db_con: Type[DbConnect]):
        """Initialize."""
        self._db_con = db_con
        self._load_refs()

    def _load_refs(self):
        """Supply mappings in format {name: id}."""
        #
        df_task = self._db_con.fetch_all(
            "select * from ref_task", ["task_id", "task_name"]
        )
        self.ref_task = {
            y: x for x, y in zip(df_task["task_id"], df_task["task_name"])
        }

        #
        df_vox = self._db_con.fetch_all(
            "select * from ref_voxel_gm", ["voxel_id", "voxel_name"]
        )
        self.ref_voxel_gm = {
            y: x for x, y in zip(df_vox["voxel_id"], df_vox["voxel_name"])
        }

    def voxel_label(self, voxel_name) -> int:
        """Return voxel ID given voxel name."""
        return self.ref_voxel_gm[voxel_name]


# %%
class MysqlUpdate(_RefMaps):
    """Title.

    Methods
    -------
    update_db(*args)
        Update appropriate table given args

    Example
    -------
    db_con = sql_database.DbConnect()
    up_mysql = sql_database.MysqlUpdate(db_con)
    up_mysql.update_db(*args)

    """

    def __init__(self, db_con: Type[DbConnect]):
        """Initialize."""
        super().__init__(db_con)

    def update_db(
        self,
        df,
        subj,
        task,
        model,
        con,
        subj_col="subj_id",
    ):
        """Title.

        Parameters
        ----------

        """
        #

        #
        self._df = df
        self._subj = subj
        self._task = task
        self._subj_col = subj_col
        self._tbl_name = f"tbl_betas_{model}_{con}_gm"

        #
        print(
            f"\tUpdating db_emorep {self._tbl_name} for {self._subj}, {task}"
        )
        if model == "lss":
            self._update_lss_betas()
        else:
            self._update_reg_betas()

    def _update_reg_betas(self):
        """Title."""
        # Add id columns
        self._df[self._subj_col] = int(self._subj.split("-ER")[-1])
        self._df["task_id"] = self.ref_task[self._task.split("-")[-1]]
        self._df["voxel_id"] = self._df.apply(
            lambda x: self.voxel_label(x.voxel_name), axis=1
        )

        #
        id_list = [
            self._subj_col,
            "task_id",
            "num_exposure",
            "voxel_id",
        ]
        emo_list = [x for x in self._df.columns if "emo" in x]
        all_cols = id_list + emo_list

        #
        val_list = []
        val_list += ["%s" for x in all_cols]
        tbl_input = list(self._df[all_cols].itertuples(index=False, name=None))

        #
        sql_cmd = (
            f"insert ignore into {self._tbl_name} ({', '.join(all_cols)}) "
            + f"values ({', '.join(val_list)})"
        )
        self._db_con.exec_many(
            sql_cmd,
            tbl_input,
        )

    def _update_lss_betas(self):
        """Title."""
        pass


# %%
