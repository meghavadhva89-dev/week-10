import pandas as pd
from apputil import predict_rating


def main():
    df_X = pd.DataFrame(
        [[10.00, "Dark"], [15.00, "Very Light"], [8.50, None]],
        columns=["100g_USD", "roast"],
    )

    preds = predict_rating(df_X)
    print("Input:\n", df_X)
    print("Predictions:\n", preds)

    # Test text prediction (Bonus Exercise 4)
    X_text = pd.DataFrame(
        [
            "A delightful coffee with hints of chocolate and caramel.",
            "A strong coffee with a bold flavor and a smoky finish.",
        ],
        columns=["text"],
    )
    try:
        preds_text = predict_rating(X_text, text=True)
        print("\nText input:\n", X_text)
        print("Text predictions:\n", preds_text)
    except Exception as e:
        print("\nText prediction skipped or failed:", e)


if __name__ == "__main__":
    main()
