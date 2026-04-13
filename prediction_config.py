import sqlite3
import pandas as pd

def get():
    MATERIAL_GROUPS_TIME_SERIES_QUERY = """
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
            coalesce(sum(s.sale_m3), 0)                as sales_m3
        from periods p
        
        cross join materials
        inner join eligible_groups
        on materials.brand = eligible_groups.brand and materials.category = eligible_groups.category
        
        left outer join sales s
        on s.year = p.year and s.month = p.month and s.material_id = materials.id
        
        group by materials.brand, materials.category, p.year, p.month
        order by materials.brand, materials.category, p.year, p.month
        """

    conn = sqlite3.connect("timber.sqlite")
    data_frame = pd.read_sql_query(MATERIAL_GROUPS_TIME_SERIES_QUERY, conn)
    conn.close()

    groups = {}
    for material_group, group_data_frame in data_frame.groupby("material_group"):
        series = (
            group_data_frame
            .sort_values("period")
            .set_index("period")["sales_m3"]
            .astype("float64")
        )
        groups[material_group] = series


    splits = [
        ("2025-07", "2025-12", "2026-01"),
        ("2025-07", "2026-01", "2026-02"),
        ("2025-07", "2026-02", "2026-03"),
    ]
    return groups, splits