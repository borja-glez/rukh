import duckdb

GAMES = "data/uci/year=2025/month=01/games.parquet"
con = duckdb.connect()

print("== Distribución de Elo (blancas), tramos de 100 ==")
print(
    con.sql(f"""
    SELECT (white_elo // 100) * 100 AS tramo,
           count(*)                  AS partidas,
           round(100.0 * count(*) / sum(count(*)) OVER (), 1) AS pct
    FROM read_parquet('{GAMES}')
    GROUP BY tramo
    ORDER BY tramo
    """)
)

print("== Duración en plies ==")
print(
    con.sql(f"""
    SELECT count(*)                            AS partidas,
           round(avg(n_plies), 1)              AS media,
           quantile_cont(n_plies, 0.5)         AS p50,
           quantile_cont(n_plies, 0.95)        AS p95,
           max(n_plies)                        AS max,
           round(100.0 * avg(CAST(n_plies <= 195 AS DOUBLE)), 1) AS pct_cabe_en_200
    FROM read_parquet('{GAMES}')
    """)
)

print("== Aperturas más frecuentes (código ECO) ==")
print(
    con.sql(f"""
    SELECT eco, count(*) AS partidas,
           round(100.0 * avg(CAST(result = '1-0' AS DOUBLE)), 1) AS pct_blancas
    FROM read_parquet('{GAMES}')
    GROUP BY eco
    ORDER BY partidas DESC
    LIMIT 10
    """)
)
