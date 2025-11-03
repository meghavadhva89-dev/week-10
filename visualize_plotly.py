"""Interactive Plotly visualizations for the Week 10 coffee dataset.

This module contains small interactive figures to explore price, rating and
roast relationships.
"""

import pandas as pd
import plotly.express as px

DATA_URL = (
    "https://raw.githubusercontent.com/leontoddjohnson/datasets/"
    "refs/heads/main/data/coffee_analysis.csv"
)


def main():
    df = pd.read_csv(DATA_URL)
    df["100g_USD"] = pd.to_numeric(df["100g_USD"], errors="coerce")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    # Scatter: price vs rating colored by roast
    fig = px.scatter(
        df,
        x="100g_USD",
        y="rating",
        color="roast",
        hover_data=[col for col in ["brand", "country"] if col in df.columns],
        title="100g_USD vs rating (colored by roast)",
        labels={"100g_USD": "100g price (USD)", "rating": "rating"},
        height=600,
    )
    fig.update_traces(marker=dict(opacity=0.75, size=8))
    # also save an HTML copy so you can open it in a browser
    fig.write_html("visual_price_vs_rating.html", include_plotlyjs="cdn")
    fig.show()

    # Box plot: rating distribution by roast
    if "roast" in df.columns:
        fig2 = px.box(df, x="roast", y="rating", points="all", title="Rating by roast")
        fig2.update_layout(xaxis_title="roast", yaxis_title="rating", height=500)
        # save HTML copy
        fig2.write_html("visual_rating_by_roast.html", include_plotlyjs="cdn")
        fig2.show()
    else:
        print("Column 'roast' not found; skipping box plot.")


if __name__ == "__main__":
    main()
