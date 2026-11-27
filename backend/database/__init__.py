"""The backend's own database code; the dataset catalogue's tables and engine
come from BIDS-SQL's package `db`.

    app_tables  the app's own tables (query_log, search history, …)
    worker      database access for code running in a worker thread
"""
