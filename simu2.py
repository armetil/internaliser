import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from collections import defaultdict, deque
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_sample_orders(n=20, seed=42, qty_min=1, qty_max=10):
    """
    Generate a sample DataFrame of n orders with columns:
      - order_id
      - side (BUY or SELL)
      - instrument (e.g., 'AAPL', 'MSFT')
      - time (datetime)
      - quantity
    """
    np.random.seed(seed)

    sides = np.random.choice(['BUY', 'SELL'], size=n)
    instruments = np.random.choice(['AAPL', 'MSFT', 'GOOG'], size=n)

    start_time = datetime(2025, 1, 1, 9, 30, 0)
    times = [start_time + timedelta(minutes=i) for i in range(n)]

    quantities = np.random.randint(qty_min, qty_max, size=n)

    df = pd.DataFrame({
        'order_id': range(1, n + 1),
        'side': sides,
        'instrument': instruments,
        'time': times,
        'quantity': quantities
    })
    return df


def simulate_internalizer_advanced(orders_df, time_window=timedelta(minutes=2)):
    """
    Simulates an internal matching engine using a dictionary of deques for unmatched orders.

    - We create a deque for each (instrument, side) pair.
    - For each new order, we:
        1) Remove expired orders from the *opposite side* deque (front) if they fall outside time_window.
        2) Match as many shares as possible with the front of the *opposite side* deque (FIFO).
        3) If the new order still has leftover quantity, we place it in its own side's deque.

    Returns an augmented DataFrame with:
      - matched_quantity
      - remaining_quantity
    """

    # Sort by arrival time
    orders_df = orders_df.sort_values(by='time').reset_index(drop=True)
    n = len(orders_df)

    matched_quantity = [0] * n
    remaining_quantity = list(orders_df['quantity'])

    # We will store unmatched orders in a dict:
    #   unmatched[(instrument, side)] = deque of tuples (idx, order_time, quantity_remaining)
    unmatched = defaultdict(deque)

    def remove_expired_from_front(q: deque, current_time: datetime):
        """
        Pop from the front of the deque while the front-most order is
        older than (current_time - time_window) or has 0 quantity left.
        """
        while q:
            _, front_time, front_qty = q[0]

            # If the front is expired or has no qty, pop it
            if front_time < current_time - time_window or front_qty <= 0:
                q.popleft()
            else:
                # The first order in the deque is neither expired nor empty,
                # so the rest are definitely not expired, because they're even newer.
                break

    for i, row in orders_df.iterrows():
        current_time = row['time']
        current_side = row['side']
        instr = row['instrument']
        current_qty = remaining_quantity[i]

        # Determine the opposite side
        opposite_side = 'SELL' if current_side == 'BUY' else 'BUY'

        # 1) Remove expired orders from the *opposite side* for this instrument
        opp_key = (instr, opposite_side)
        remove_expired_from_front(unmatched[opp_key], current_time)

        # 2) Match as many shares as possible with the front of the opposite side's deque
        while current_qty > 0 and unmatched[opp_key]:
            old_idx, old_time, old_qty = unmatched[opp_key][0]

            # Check if it's now expired (in case it was borderline)
            if old_time < current_time - time_window or old_qty <= 0:
                unmatched[opp_key].popleft()
                continue

            matchable_qty = min(current_qty, old_qty)
            matched_quantity[i] += matchable_qty
            matched_quantity[old_idx] += matchable_qty

            current_qty -= matchable_qty
            old_qty -= matchable_qty

            # Update the old unmatched order
            if old_qty == 0:
                unmatched[opp_key].popleft()
            else:
                # Update the tuple in place
                unmatched[opp_key][0] = (old_idx, old_time, old_qty)

        # 3) If there's leftover, put the new order in its own side's deque
        remaining_quantity[i] = current_qty
        if current_qty > 0:
            my_key = (instr, current_side)
            unmatched[my_key].append((i, current_time, current_qty))

    # Build result
    result_df = orders_df.copy()
    result_df['matched_quantity'] = matched_quantity
    result_df['remaining_quantity'] = remaining_quantity
    return result_df


def compute_internalization_rate(matched_df):
    total_shares = matched_df['quantity'].sum()
    total_matched = matched_df['matched_quantity'].sum()
    return (total_matched / total_shares) if total_shares else 0.0


def compute_internalization_rate_by_instrument(matched_df):
    """
    Computes the internalization rate by instrument:
      sum(matched_quantity) / sum(quantity)  for each instrument
    Returns a DataFrame with columns: total_quantity, total_matched, rate
    """
    group = matched_df.groupby('instrument').agg(
        total_quantity=('quantity', 'sum'),
        total_matched=('matched_quantity', 'sum')
    )
    group['internalization_rate'] = group['total_matched'] / group['total_quantity']

    group['internalization_rate'] = (group['internalization_rate'] * 100).round(2).astype(str) + '%'
    return group


def plot_evolution_of_internalization_rate(matched_df, rolling_window=1000):
    """
    Plots the evolution of the internalization rate over time:
      1) Overall cumulative rate vs. time
      2) Cumulative rate by instrument vs. time (multiple lines)

    We'll:
      - Sort by time.
      - Compute cumulative sums of quantity and matched_quantity.
      - Plot ratio = cumulative_matched / cumulative_quantity as a function of time.
    """

    # Sort by time
    df_sorted = matched_df.sort_values('time').copy()

    # -------------------------
    # 1) Overall Cumulative Rate
    # -------------------------
    df_sorted['cumulative_quantity'] = df_sorted['quantity'].cumsum()
    df_sorted['cumulative_matched'] = df_sorted['matched_quantity'].cumsum()
    df_sorted['cumulative_rate'] = df_sorted['cumulative_matched'] / df_sorted['cumulative_quantity']

    df_sorted['rolling_quantity'] = df_sorted['quantity'].rolling(window=rolling_window, min_periods=1).sum()
    df_sorted['rolling_matched'] = df_sorted['matched_quantity'].rolling(window=rolling_window, min_periods=1).sum()
    df_sorted['rolling_rate'] = df_sorted['rolling_matched'] / df_sorted['rolling_quantity']

    fig = plt.figure(figsize=(9, 4))
    fig.suptitle('Cumulative Internalization Rate Over Time')
    ax, ax2 = fig.subplots(2, 1, sharex=True, sharey=True)
    ax.plot(df_sorted['time'], df_sorted['cumulative_rate'], label='cumsum')
    ax.plot(df_sorted['time'], df_sorted['rolling_rate'], label=f'rolling[{rolling_window}]')
    ax.set_title('Overall')
    ax.set_ylabel('Rate')
    ax.grid(True)
    ax.legend()

    # -----------------------------------
    # 2) Cumulative Rate By Instrument
    # -----------------------------------
    # We group by instrument but still plot as a time series,
    # so we do cumulative sums *per instrument* in time order.
    df_by_instr = df_sorted.copy()
    df_by_instr['cumulative_qty_instr'] = df_by_instr.groupby('instrument')['quantity'].cumsum()
    df_by_instr['cumulative_match_instr'] = df_by_instr.groupby('instrument')['matched_quantity'].cumsum()
    df_by_instr['cumulative_rate_instr'] = (
            df_by_instr['cumulative_match_instr'] / df_by_instr['cumulative_qty_instr']
    )

    for instr in sorted(df_by_instr['instrument'].unique()):
        subset = df_by_instr[df_by_instr['instrument'] == instr]
        # Plot each instrument's line on the same figure
        ax2.plot(subset['time'], subset['cumulative_rate_instr'], label=instr)

    ax2.set_xlabel('Time')
    ax2.set_ylabel('Rate')
    ax2.set_title('By instrument')
    ax2.grid(True)
    ax2.legend()

    plt.show()


def plot_evolution_of_internalization_rate_plotly_combined(matched_df, rolling_window=1000):
    """
    Plot the evolution of the internalization rate using Plotly:
      1) Overall cumulative vs. rolling rate
      2) Cumulative rate by instrument

    Parameters:
        matched_df (pd.DataFrame): Input data with 'time', 'quantity', 'matched_quantity', 'instrument'
        rolling_window (int): Window size for rolling rate
    """

    # Sort and compute necessary columns
    df_sorted = matched_df.sort_values('time').copy()
    df_sorted['cumulative_quantity'] = df_sorted['quantity'].cumsum()
    df_sorted['cumulative_matched'] = df_sorted['matched_quantity'].cumsum()
    df_sorted['cumulative_rate'] = df_sorted['cumulative_matched'] / df_sorted['cumulative_quantity']

    df_sorted['rolling_quantity'] = df_sorted['quantity'].rolling(window=rolling_window, min_periods=1).sum()
    df_sorted['rolling_matched'] = df_sorted['matched_quantity'].rolling(window=rolling_window, min_periods=1).sum()
    df_sorted['rolling_rate'] = df_sorted['rolling_matched'] / df_sorted['rolling_quantity']

    # Per instrument cumulative
    df_by_instr = df_sorted.copy()
    df_by_instr['cumulative_qty_instr'] = df_by_instr.groupby('instrument')['quantity'].cumsum()
    df_by_instr['cumulative_match_instr'] = df_by_instr.groupby('instrument')['matched_quantity'].cumsum()
    df_by_instr['cumulative_rate_instr'] = (
        df_by_instr['cumulative_match_instr'] / df_by_instr['cumulative_qty_instr']
    )
    df_roll_by_instr = df_sorted.copy()
    df_roll_by_instr['rolling_quantity_instr'] = (
        df_roll_by_instr.groupby('instrument')['quantity']
            .rolling(window=rolling_window, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
    )

    df_roll_by_instr['rolling_matched_instr'] = (
        df_roll_by_instr.groupby('instrument')['matched_quantity']
            .rolling(window=rolling_window, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
    )

    df_roll_by_instr['rolling_rate_instr'] = (
            df_roll_by_instr['rolling_matched_instr'] / df_roll_by_instr['rolling_quantity_instr']
    )
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=["Overall Internalization Rate", "Cumulative Rate by Instrument"]
    )

    # Plot cumulative and rolling rate
    fig.add_trace(go.Scatter(
        x=df_sorted['time'], y=df_sorted['cumulative_rate'],
        mode='lines', name='Cumulative Rate'), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df_sorted['time'], y=df_sorted['rolling_rate'],
        mode='lines', name=f'Rolling Rate ({rolling_window})'), row=1, col=1)

    # Plot per-instrument cumulative rates
    for instr in sorted(df_by_instr['instrument'].unique()):
        subset = df_by_instr[df_by_instr['instrument'] == instr]
        rolling_subset = df_roll_by_instr[df_roll_by_instr['instrument'] == instr]

        fig.add_trace(go.Scatter(
            x=subset['time'], y=subset['cumulative_rate_instr'],
            mode='lines', name=f'Instr: {instr}'), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=rolling_subset['time'], y=rolling_subset['rolling_rate_instr'],
            mode='lines', name=f'Instr(rolling={rolling_window}): {instr}'), row=2, col=1)

    # Update layout
    fig.update_layout(
        height=800, width=1000,
        title_text='Internalization Rate Over Time (Plotly)',
        showlegend=True
    )
    fig.update_xaxes(title_text='Time', row=2, col=1)
    fig.update_yaxes(title_text='Rate', row=1, col=1)
    fig.update_yaxes(title_text='Rate', row=2, col=1)

    fig.show()

# Simple demonstration
if __name__ == "__main__":
    df_orders = generate_sample_orders(n=200000, seed=42)
    matched_df = simulate_internalizer_advanced(df_orders, time_window=timedelta(minutes=2))
    rate = compute_internalization_rate(matched_df)
    rate_dct = compute_internalization_rate_by_instrument(matched_df)

    print("=== Orders ===")
    print(df_orders)
    print("\n=== Matched ===")
    print(matched_df)
    print(f"\nOverall Internalization Rate: {rate:.2%}")

    print("\nInternalization Rate by Instrument:")
    print(rate_dct)

    plot_evolution_of_internalization_rate_plotly_combined(matched_df)
    # plot_evolution_of_internalization_rate(matched_df)
