import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_sample_orders(n=20, seed=42, qty_min=0, qty_max=20):
    """
    Generate a sample DataFrame of n orders with columns:
      - order_id
      - side (BUY or SELL)
      - instrument (e.g., 'AAPL', 'MSFT')
      - time (datetime)
      - quantity
    """
    np.random.seed(seed)

    # Randomly pick sides
    sides = np.random.choice(['BUY', 'SELL'], size=n)

    # Randomly pick instruments from a small list
    instruments = np.random.choice(['AAPL', 'MSFT', 'GOOG'], size=n)

    # Generate times spaced about 1 minute apart for simplicity
    start_time = datetime(2025, 1, 1, 9, 30, 0)
    time_deltas = [timedelta(minutes=i) for i in range(n)]
    times = [start_time + td for td in time_deltas]

    # Random quantities
    quantities = np.random.randint(qty_min, qty_max, size=n)  # 10 to 199 shares

    df = pd.DataFrame({
        'order_id': range(1, n + 1),
        'side': sides,
        'instrument': instruments,
        'time': times,
        'quantity': quantities
    })

    return df


def simulate_internalizer(orders_df, time_window=timedelta(minutes=10)):
    """
    Simulates an internal matching engine that tries to match
    buy/sell orders for the same instrument within a `time_window`,
    allowing partial or full fills based on order quantity.

    Matching logic:
      - Sort orders by time (chronological).
      - Maintain a list of unmatched orders (within time_window).
      - For each new order, remove unmatched orders older than time_window.
      - Match as many shares as possible with existing unmatched orders of the
        opposite side for the same instrument (FIFO-style).
      - Track the number of shares matched internally for each order.
      - The remainder (if any) of the new order is added to unmatched orders.
      - Once the time window closes on an unmatched order, or the order is fully
        matched, it is effectively out of the unmatched pool.

    Returns an augmented DataFrame with columns:
      - matched_quantity: total shares matched internally
      - remaining_quantity: leftover shares (routed externally if not matched in time)
    """
    # Sort orders by arrival time
    orders_df = orders_df.sort_values(by='time').reset_index(drop=True)

    # For tracking unmatched orders
    # We'll keep a list of dicts with keys:
    #   { 'idx': int, 'order_id': int, 'side': str, 'instrument': str,
    #     'time': datetime, 'quantity': int (remaining unmatched qty) }
    unmatched_orders = []

    # Prepare output columns
    matched_quantity = [0] * len(orders_df)  # how many shares each order matched internally
    remaining_quantity = list(orders_df['quantity'])  # how many shares remain unmatched

    def remove_expired_orders(current_time):
        """Remove any unmatched orders that are beyond the time_window."""
        nonlocal unmatched_orders
        unmatched_orders = [
            o for o in unmatched_orders
            if o['time'] >= current_time - time_window and o['quantity'] > 0
        ]

    for i, row in orders_df.iterrows():
        current_time = row['time']
        current_side = row['side']
        current_instrument = row['instrument']
        current_qty = remaining_quantity[i]

        # Remove orders that are outside the time window
        remove_expired_orders(current_time)

        # Attempt to match with opposite side, same instrument
        # We'll do FIFO matching: earliest unmatched first
        # (You can customize priority logic as needed.)

        # We'll iterate through unmatched orders that meet criteria
        for u in unmatched_orders:
            if (
                    u['side'] != current_side and
                    u['instrument'] == current_instrument and
                    current_qty > 0
            ):
                # We can match up to min of each side's quantity
                matchable_qty = min(current_qty, u['quantity'])

                # Adjust matched quantities
                matched_quantity[i] += matchable_qty
                matched_quantity[u['idx']] += matchable_qty

                # Update both leftover quantities
                current_qty -= matchable_qty
                u['quantity'] -= matchable_qty

        # Update the leftover quantity in the main record
        remaining_quantity[i] = current_qty

        # If there's still some quantity left, add it to unmatched orders
        if current_qty > 0:
            unmatched_orders.append({
                'idx': i,
                'order_id': row['order_id'],
                'side': current_side,
                'instrument': current_instrument,
                'time': current_time,
                'quantity': current_qty
            })

    # Construct result DataFrame
    result_df = orders_df.copy()
    result_df['matched_quantity'] = matched_quantity
    result_df['remaining_quantity'] = remaining_quantity

    return result_df


def compute_internalization_rate(matched_df):
    """
    Computes the internalization rate by *shares*, i.e.
    total matched shares / total shares in all orders.
    """
    total_shares = matched_df['quantity'].sum()
    total_matched = matched_df['matched_quantity'].sum()  # sum of matched shares for all orders
    if total_shares == 0:
        return 0.0
    return total_matched / total_shares


if __name__ == "__main__":
    # 1) Generate sample data
    n = 200000
    minutes = 2
    df_orders = generate_sample_orders(n=n)

    # 2) Simulate internalizer with quantity-based partial matching
    matched_results = simulate_internalizer(df_orders, time_window=timedelta(minutes=minutes))

    # 3) Compute internalization rate (by shares)
    internal_rate = compute_internalization_rate(matched_results)

    print(f"{n=}; {minutes=}")
    print("=== Orders Data ===")
    print(df_orders)
    print("\n=== Matched Results ===")
    print(matched_results)
    print(f"\nInternalization Rate (by shares): {internal_rate:.2%}")
