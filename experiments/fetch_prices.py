"""Pull full daily price history for the universe from the public
defeatbeta/yahoo-finance-data dataset on Hugging Face into <out_dir>/<SYM>.csv.

    uv run --with duckdb python experiments/fetch_prices.py data_long
"""
import duckdb, sys
out=sys.argv[1]
B="https://huggingface.co/datasets/defeatbeta/yahoo-finance-data/resolve/main/data/US/stock_prices.parquet"
U="AAPL ABBV ABT ADBE AMD AMGN AMZN AVGO AXP BA BAC BLK BMY C CAT CMCSA COST CRM CSCO CVX DHR DIS GE GILD GOOGL GS HD HON IBM INTC JNJ JPM KO LIN LLY LOW MA MCD MDT META MRK MS MSFT NFLX NKE NOW NVDA ORCL PEP PFE PG PM QCOM SBUX SPY T TMO TSLA TXN UNH UNP V VZ WFC WMT".split()
c=duckdb.connect()
c.sql(f"create table p as select symbol,report_date as date,open::double open,high::double high,low::double low,close::double close,volume::double volume from '{B}' where symbol in {tuple(U)}")
for s in U:
    c.sql(f"copy (select date,open,high,low,close,volume from p where symbol='{s}' order by date) to '{out}/{s}.csv' (header)")
print(c.sql("select count(distinct symbol),min(date),max(date),count(*) from p").fetchall())
print("latest first-date:",c.sql("select symbol,min(date) d from p group by 1 order by d desc limit 4").fetchall())
print("AAPL around 2020 split:",c.sql("select date,close from p where symbol='AAPL' and date between '2020-08-27' and '2020-09-01' order by 1").fetchall())
print("dupes:",c.sql("select count(*) from (select symbol,date from p group by 1,2 having count(*)>1)").fetchall(), "nulls/nonpos:",c.sql("select count(*) from p where close is null or close<=0 or open is null or open<=0").fetchall())
