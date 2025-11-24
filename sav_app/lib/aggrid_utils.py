import pandas as pd

try:
    from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode, JsCode
    AGGRID_OK = True
except Exception:
    AGGRID_OK = False
    AgGrid = GridOptionsBuilder = GridUpdateMode = DataReturnMode = JsCode = None  # type: ignore

def aggrid_selected_rows(grid_resp):
    sel = getattr(grid_resp, "selected_rows", None)
    if sel is not None:
        if isinstance(sel, pd.DataFrame):
            return sel.to_dict("records")
        if isinstance(sel, list):
            return sel
    if isinstance(grid_resp, dict):
        sel = grid_resp.get("selected_rows")
        if isinstance(sel, pd.DataFrame):
            return sel.to_dict("records")
        if isinstance(sel, list):
            return sel
    return []
