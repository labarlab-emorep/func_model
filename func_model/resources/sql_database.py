"""Methods for interacting with mysql db_emorep.

DbConnect : connect to and interact with db_emorep on mysql server
DbUpdateBetas : update db_emorep tables

"""

# %%
import os
import platform
import pandas as pd
import numpy as np
from typing import Type
from contextlib import contextmanager

if "labarserv2" in platform.uname().node:
    import mysql.connector
elif "dcc" in platform.uname().node:
    import pymysql
    import paramiko
    from sshtunnel import SSHTunnelForwarder


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
    fetch_df()
        Return pd.DataFrame from query statement
    fetch_rows()
        Return rows from query statement

    Notes
    -----
    Requires environment variable 'SQL_PASS' to contain user password
    for mysql db_emorep.

    Example
    -------
    db_con = DbConnect()
    row = db_con.fetch_rows("select * from ref_subj limit 1")
    db_con.close_con()

    """

    def __init__(self):
        """Set con attr as mysql connection."""
        try:
            os.environ["SQL_PASS"]
        except KeyError as e:
            raise Exception(
                "No global variable 'SQL_PASS' defined in user env"
            ) from e

        if "labarserv2" in platform.uname().node:
            self._connect_ls2()
        elif "dcc" in platform.uname().node:
            self._connect_dcc()

    def _connect_ls2(self):
        """Connect to MySQL server from labarserv2."""
        self.con = mysql.connector.connect(
            host="localhost",
            user=os.environ["USER"],
            password=os.environ["SQL_PASS"],
            database="db_emorep",
        )

    def _connect_dcc(self):
        """Connect to MySQL server from DCC."""
        try:
            os.environ["RSA_LS2"]
        except KeyError as e:
            raise Exception(
                "No global variable 'RSA_LS2' defined in user env"
            ) from e

        self._connect_ssh()
        self.con = pymysql.connect(
            host="127.0.0.1",
            user=os.environ["USER"],
            passwd=os.environ["SQL_PASS"],
            db="db_emorep",
            port=self._ssh_tunnel.local_bind_port,
        )

    def _connect_ssh(self):
        """Start ssh tunnel."""
        rsa_keoki = paramiko.RSAKey.from_private_key_file(
            os.environ["RSA_LS2"]
        )
        self._ssh_tunnel = SSHTunnelForwarder(
            ("ccn-labarserv2.vm.duke.edu", 22),
            ssh_username=os.environ["USER"],
            ssh_pkey=rsa_keoki,
            remote_bind_address=("127.0.0.1", 3306),
        )
        self._ssh_tunnel.start()

    @contextmanager
    def _con_cursor(self):
        """Yield cursor."""
        cursor = self.con.cursor()
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

    def fetch_df(self, sql_cmd: str, col_names: list) -> pd.DataFrame:
        """Return dataframe from query output.

        Example
        -------
        db_con = sql_database.DbConnect()
        sql_cmd = "select * from ref_subj"
        col_names = ["subj_id", "subj_name"]
        df_subj = db_con.fetch_df(sql_cmd, col_names)

        """
        return pd.DataFrame(self.fetch_rows(sql_cmd), columns=col_names)

    def fetch_rows(self, sql_cmd: str) -> list:
        """Return rows from query output.

        Example
        -------
        db_con = sql_database.DbConnect()
        sql_cmd = "select * from ref_subj"
        rows = db_con.fetch_df(sql_cmd)

        """
        with self._con_cursor() as cur:
            cur.execute(sql_cmd)
            rows = cur.fetchall()
        return rows

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
        df_task = self._db_con.fetch_df(
            "select * from ref_task", ["task_id", "task_name"]
        )
        self.ref_task = {
            y: x for x, y in zip(df_task["task_id"], df_task["task_name"])
        }

        # Reference voxels
        df_vox = self._db_con.fetch_df(
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
            + f"where task_id = {task_id} and subj_id = {subj_id} "
            + "limit 1"
        )
        row = self._db_con.fetch_rows(sql_cmd)
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
        tbl_name = f"tbl_betas_{model}_{con}_gm"
        print(f"\tUpdating db_emorep {tbl_name} for {subj}, {task}")

        # Add id columns
        df["subj_id"] = int(subj.split("-ER")[-1])
        df["task_id"] = self.ref_task[task.split("-")[-1]]
        df["voxel_id"] = df.apply(
            lambda x: self.voxel_label(x.voxel_name), axis=1
        )

        # Manage NaNs
        df = df.replace({np.nan: None})

        # Determine relevant columns for table
        id_list = self._id_cols(model)
        emo_list = [x for x in df.columns if "emo" in x]
        all_cols = id_list + emo_list

        # Build input data and insert command
        val_list = ["%s" for x in all_cols]
        tbl_input = list(df[all_cols].itertuples(index=False, name=None))
        sql_cmd = (
            f"insert into {tbl_name} ({', '.join(all_cols)}) "
            + f"values ({', '.join(val_list)})"
        )

        # Manage overwrite request, update table
        if overwrite:
            vals = [f"{x}=values({x})" for x in emo_list]
            up_cmd = f" on duplicate key update {', '.join(vals)}"
            sql_cmd = sql_cmd + up_cmd
        self._db_con.exec_many(
            sql_cmd,
            tbl_input,
        )

    def _id_cols(self, model: str) -> list:
        """Return list of primary key columns."""
        if model == "lss":
            return [
                "subj_id",
                "task_id",
                "num_block",
                "num_event",
                "voxel_id",
            ]
        else:
            return [
                "subj_id",
                "task_id",
                "num_block",
                "voxel_id",
            ]


# %%
