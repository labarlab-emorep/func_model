"""Methods for interacting with mysql db_emorep.

DbConnect : connect to and interact with db_emorep on mysql server
DbUpdateBetas : update db_emorep tables

"""
# %%
import os
import pandas as pd
from typing import Type, Tuple
import mysql.connector
from contextlib import contextmanager


# %%
class DbConnect:
    """Supply db_emorep database connection and interaction methods.

    Attributes
    ----------
    con : mysql.connector.connection_cext.CMySQLConnection
        Connection object to database

    Methods
    -------
    close_con()
        Close database connection
    exec_many()
        Update mysql db_emorep.tbl_* with multiple values
    fetch_all()
        Fetch all records given a query statement
    fetch_one()
        Fetch one record given a query statement

    Notes
    -----
    Requires environment variable 'SQL_PASS' to contain user password
    for mysql db_emorep.

    Example
    -------
    db_con = DbConnect()
    row = db_con.fetch_one("select * from ref_subj")
    db_con.close_con()

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
    def _con_cursor(self, buf: int = False):
        """Yield cursor."""
        cursor = self.con.cursor(buffered=buf)
        try:
            yield cursor
        finally:
            cursor.close()

    def exec_many(self, sql_cmd: str, value_list: list):
        """Update db_emorep via executemany.

        Example
        -------
        db_con = sql_database.DbConnect()
        sql_cmd = (
            "insert ignore into ref_subj (subj_id, subj_name) values (%s, %s)"
        )
        tbl_input = [(9, "ER0009"), (16, "ER0016")]
        db_con.exec_many(sql_cmd, tbl_input)

        """
        with self._con_cursor() as cur:
            cur.executemany(sql_cmd, value_list)
            self.con.commit()

    def fetch_all(self, sql_cmd: str, col_names: list) -> pd.DataFrame:
        """Return dataframe from query output.

        Example
        -------
        db_con = sql_database.DbConnect()
        sql_cmd = "select * from ref_subj"
        col_names = ["subj_id", "subj_name"]
        df_subj = db_con.fetch_all(sql_cmd, col_names)

        """
        with self._con_cursor() as cur:
            cur.execute(sql_cmd)
            df = pd.DataFrame(cur.fetchall(), columns=col_names)
        return df

    def fetch_one(self, sql_cmd: str) -> Tuple:
        """Return single row from query output.

        Example
        -------
        db_con = sql_database.DbConnect()
        sql_cmd = "select * from ref_subj"
        row = db_con.fetch_all(sql_cmd)

        """
        with self._con_cursor(buf=True) as cur:
            cur.execute(sql_cmd)
            row = cur.fetchone()
        return row

    def close_con(self):
        """Close database connection."""
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
        # Reference task
        df_task = self._db_con.fetch_all(
            "select * from ref_task", ["task_id", "task_name"]
        )
        self.ref_task = {
            y: x for x, y in zip(df_task["task_id"], df_task["task_name"])
        }

        # Reference voxels
        df_vox = self._db_con.fetch_all(
            "select * from ref_voxel_gm", ["voxel_id", "voxel_name"]
        )
        self.ref_voxel_gm = {
            y: x for x, y in zip(df_vox["voxel_id"], df_vox["voxel_name"])
        }

    def voxel_label(self, voxel_name: str) -> int:
        """Return voxel ID given voxel name."""
        return self.ref_voxel_gm[voxel_name]


# %%
class DbUpdateBetas(_RefMaps):
    """Check and update beta tables in db_emorep.

    Parameters
    ----------
    db_con : func_model.resources.general.sql_database.DbConnect
        Database connection instance

    Methods
    -------
    check_db()
        Check if one record exists in beta table
    update_db()
        Prep pd.DataFrame and then update beta table

    """

    def __init__(self, db_con: Type[DbConnect]):
        """Initialize."""
        super().__init__(db_con)

    def check_db(self, subj: str, task: str, model: str, con: str) -> bool:
        """Check if beta table already has subject data.

        Example
        -------
        db_con = sql_database.DbConnect()
        update_betas = sql_database.DbUpdateBetas(db_con)
        row_exists = update_betas.check_db(
            "sub-ER0009", "task-movies", "sep", "stim"
        )

        """
        subj_id = int(subj.split("-ER")[-1])
        task_id = self.ref_task[task.split("-")[-1]]
        sql_cmd = (
            f"select * from tbl_betas_{model}_{con}_gm "
            + f"where task_id = {task_id} and subj_id = {subj_id}"
        )
        row = self._db_con.fetch_one(sql_cmd)
        return True if row else False

    def update_db(
        self,
        df: pd.DataFrame,
        subj: str,
        task: str,
        model: str,
        con: str,
        overwrite: bool,
    ):
        """Update beta table from pd.DataFrame.

        Example
        -------
        db_con = sql_database.DbConnect()
        update_betas = sql_database.DbUpdateBetas(db_con)
        update_betas.update_db(
            pd.DataFrame,
            "sub-ER0009",
            "task-movies",
            "sep",
            "stim",
            False,
        )

        """
        self._df = df
        self._subj = subj
        self._task = task
        self._overwrite = overwrite
        self._subj_col = "subj_id"
        self._tbl_name = f"tbl_betas_{model}_{con}_gm"

        print(
            f"\tUpdating db_emorep {self._tbl_name} for {self._subj}, {task}"
        )
        if model == "lss":
            self._update_lss_betas()
        else:
            self._update_reg_betas()

    def _update_reg_betas(self):
        """Prep pd.DataFrame and then insert records."""
        # Add id columns
        self._df[self._subj_col] = int(self._subj.split("-ER")[-1])
        self._df["task_id"] = self.ref_task[self._task.split("-")[-1]]
        self._df["voxel_id"] = self._df.apply(
            lambda x: self.voxel_label(x.voxel_name), axis=1
        )

        # Determine relevant columns for table
        id_list = [
            self._subj_col,
            "task_id",
            "num_exposure",
            "voxel_id",
        ]
        emo_list = [x for x in self._df.columns if "emo" in x]
        all_cols = id_list + emo_list

        # Build input data and insert command
        val_list = ["%s" for x in all_cols]
        tbl_input = list(self._df[all_cols].itertuples(index=False, name=None))
        sql_cmd = (
            f"insert into {self._tbl_name} ({', '.join(all_cols)}) "
            + f"values ({', '.join(val_list)})"
        )

        # Manage overwrite request, update table
        if self._overwrite:
            vals = [f"{x}=values({x})" for x in emo_list]
            up_cmd = f" on duplicate key update {', '.join(vals)}"
            sql_cmd = sql_cmd + up_cmd
        self._db_con.exec_many(
            sql_cmd,
            tbl_input,
        )

    def _update_lss_betas(self):
        """Title."""
        pass


# %%
