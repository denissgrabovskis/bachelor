import sqlite3
import pandas as pd
from functools import reduce

def get_groups_with_summed(*fields, transform = None):
    conn = sqlite3.connect("timber.sqlite")
    data_frames = []

    if len(fields) == 0: raise Exception()

    for field in fields:
        table,column = field.split('.')

        GROUPS_WITH_SUMMED_COLUMN_QUERY = f"""
            with
                eligible_groups as (
                    select
                        materials.brand as brand,
                        materials.category as category,
                        count(distinct s.year||s.month) as months_with_sales -- count distinct periods
                    from sales s
            
                    join materials
                    on s.material_id = materials.id
                        
                    where coalesce(s.sale_m3, 0) > 0 -- count only periods with non-zero sales
            
                    group by materials.brand, materials.category
                    having 
                        months_with_sales >= 6 -- count where periods exceed at least 6
                        and not (materials.brand = 'KARTEX' and materials.category = 'DAŽĀDAS PRECES') -- exclude specific KARTEX: DAŽĀDAS PRECES
                ),
                periods as (select distinct year, month from sales) -- from July 2025 to March 2026
            
            select
                materials.brand||': '||materials.category  as material_group,
                printf('%d-%02d', p.year, p.month)         as period,
                coalesce(sum({field}), 0)                  as {column}
            from periods p
            
            cross join materials
            inner join eligible_groups
            on materials.brand = eligible_groups.brand and materials.category = eligible_groups.category
            
            left outer join {table}
            on {table}.year = p.year and {table}.month = p.month and {table}.material_id = materials.id
            
            group by materials.brand, materials.category, p.year, p.month
            order by materials.brand, materials.category, p.year, p.month
            """

        data_frames.append(pd.read_sql_query(GROUPS_WITH_SUMMED_COLUMN_QUERY, conn))


    data_frame = reduce(lambda left, right: pd.merge(left, right, on=["material_group", "period"]), data_frames)
    conn.close()

    groups = []
    i = 0
    for group_name, group_data_frame in data_frame.groupby("material_group"):
        series = (group_data_frame
            .sort_values("period")
            .set_index("period")
            .copy())

        groups.append(transform(series, group_name, i))
        i+=1

    return groups

def get_train_splits():
    return [
        ("2025-07", "2025-12", "2026-01"),
        ("2025-07", "2026-01", "2026-02"),
        ("2025-07", "2026-02", "2026-03"),
    ]
