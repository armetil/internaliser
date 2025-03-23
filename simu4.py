import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict, deque
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_sample_orders(n=20, seed=42, qty_min=1, qty_max=10):
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

    df['side'] = df['side'].astype('category')
    df['instrument'] = df['instrument'].astype('category')
    return df


def simulate_internalizer_advanced(orders_df, time_window=timedelta(minutes=2)):
    orders_df = orders_df.sort_values(by='time').reset_index(drop=True)
    n = len(orders_df)
    matched_quantity = [0] * n
    remaining_quantity = list(orders_df['quantity'])
    unmatched = defaultdict(deque)
    match_events = []

    def remove_expired_from_front(q, current_time):
        while q and (q[0][1] < current_time - time_window or q[0][2] <= 0):
            q.popleft()

    for row in orders_df.itertuples(index=True, name=None):
        i, order_id, current_side, instr, current_time, qty = row
        current_qty = remaining_quantity[i]
        opposite_side = 'SELL' if current_side == 'BUY' else 'BUY'
        opp_key = (instr, opposite_side)

        remove_expired_from_front(unmatched[opp_key], current_time)

        while current_qty > 0 and unmatched[opp_key]:
            old_idx, old_time, old_qty = unmatched[opp_key][0]
            matchable_qty = min(current_qty, old_qty)

            matched_quantity[i] += matchable_qty
            matched_quantity[old_idx] += matchable_qty
            match_events.append({
                'buy_order_id': i if current_side == 'BUY' else old_idx,
                'sell_order_id': old_idx if current_side == 'BUY' else i,
                'matched_quantity': matchable_qty,
                'match_time': current_time,
                'buy_time': current_time if current_side == 'BUY' else old_time,
                'sell_time': old_time if current_side == 'BUY' else current_time
            })
            current_qty -= matchable_qty
            old_qty -= matchable_qty

            if old_qty == 0:
                unmatched[opp_key].popleft()
            else:
                unmatched[opp_key][0] = (old_idx, old_time, old_qty)

        remaining_quantity[i] = current_qty
        if current_qty > 0:
            unmatched[(instr, current_side)].append((i, current_time, current_qty))

    result_df = orders_df.copy()
    result_df['matched_quantity'] = matched_quantity
    result_df['remaining_quantity'] = remaining_quantity

    match_df = pd.DataFrame(match_events)
    if not match_df.empty:
        match_df['buy_time_to_match'] = (match_df['match_time'] - match_df['buy_time']).dt.total_seconds()
        match_df['sell_time_to_match'] = (match_df['match_time'] - match_df['sell_time']).dt.total_seconds()
        buy_avg = match_df.groupby('buy_order_id')['buy_time_to_match'].mean()
        sell_avg = match_df.groupby('sell_order_id')['sell_time_to_match'].mean()
        avg_time_to_match = pd.concat([buy_avg, sell_avg]).groupby(level=0).mean()
        result_df['avg_time_to_match'] = result_df.index.to_series().map(avg_time_to_match)
    else:
        result_df['avg_time_to_match'] = np.nan

    return result_df


def compute_internalization_rate(matched_df):
    total = matched_df['quantity'].sum()
    return matched_df['matched_quantity'].sum() / total if total else 0.0


def compute_internalization_rate_by_instrument(matched_df):
    group = matched_df.groupby('instrument').agg(
        total_quantity=('quantity', 'sum'),
        total_matched=('matched_quantity', 'sum')
    )
    group['internalization_rate'] = (group['total_matched'] / group['total_quantity'] * 100).round(2).astype(str) + '%'
    return group


def compute_fill_rate_stats(matched_df):
    full_fill = (matched_df['matched_quantity'] == matched_df['quantity']).mean()
    partial_fill = ((matched_df['matched_quantity'] > 0) & (matched_df['matched_quantity'] < matched_df['quantity'])).mean()
    by_instr = matched_df.groupby('instrument').apply(lambda g: pd.Series({
        'full_fill_rate': (g['matched_quantity'] == g['quantity']).mean(),
        'partial_fill_rate': ((g['matched_quantity'] > 0) & (g['matched_quantity'] < g['quantity'])).mean()
    }))
    return full_fill, partial_fill, by_instr


def plot_combined_dashboard(matched_df, rolling_window=1000):
    instruments = matched_df['instrument'].unique()
    num_rows = 4 + len(instruments)

    fig = make_subplots(
        rows=num_rows, cols=1,
        subplot_titles=(
            ["Overall Internalization Rate"] +
            ["Cumulative & Rolling Rate by Instrument"] +
            ["Distribution of Avg Time to Match"] +
            [f"Avg Time to Match - {instr}" for instr in instruments] +
            ["Match Fill Rate Over Time"]
        ),
        shared_xaxes=False
    )

    df_sorted = matched_df.sort_values('time').copy()
    df_sorted['cumulative_quantity'] = df_sorted['quantity'].cumsum()
    df_sorted['cumulative_matched'] = df_sorted['matched_quantity'].cumsum()
    df_sorted['cumulative_rate'] = df_sorted['cumulative_matched'] / df_sorted['cumulative_quantity']
    df_sorted['rolling_quantity'] = df_sorted['quantity'].rolling(window=rolling_window, min_periods=1).sum()
    df_sorted['rolling_matched'] = df_sorted['matched_quantity'].rolling(window=rolling_window, min_periods=1).sum()
    df_sorted['rolling_rate'] = df_sorted['rolling_matched'] / df_sorted['rolling_quantity']

    fig.add_trace(go.Scatter(x=df_sorted['time'], y=df_sorted['cumulative_rate'], name='Cumulative Rate'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_sorted['time'], y=df_sorted['rolling_rate'], name='Rolling Rate'), row=1, col=1)
    fig.update_yaxes(title_text='Rate', row=1, col=1)

    for instr in sorted(instruments):
        sub = df_sorted[df_sorted['instrument'] == instr].copy()
        sub['cumulative_instr'] = sub['matched_quantity'].cumsum() / sub['quantity'].cumsum()
        sub['rolling_instr'] = sub['matched_quantity'].rolling(window=rolling_window, min_periods=1).sum() / sub['quantity'].rolling(window=rolling_window, min_periods=1).sum()
        fig.add_trace(go.Scatter(x=sub['time'], y=sub['cumulative_instr'], name=f'{instr} Cumulative'), row=2, col=1)
        fig.add_trace(go.Scatter(x=sub['time'], y=sub['rolling_instr'], name=f'{instr} Rolling'), row=2, col=1)
    fig.update_yaxes(title_text='Rate', row=2, col=1)

    fig.add_trace(go.Histogram(x=matched_df['avg_time_to_match'].dropna(), nbinsx=100), row=3, col=1)

    for i, instr in enumerate(instruments):
        x = matched_df[matched_df['instrument'] == instr]['avg_time_to_match'].dropna()
        fig.add_trace(go.Histogram(x=x, nbinsx=100, name=instr), row=4 + i, col=1)
        fig.update_xaxes(title_text='Time Sec', row=4 + i, col=1)
        fig.update_yaxes(title_text='Orders', row=4 + i, col=1)

    last_row = num_rows
    matched_df['time_bucket'] = matched_df['time'].dt.floor('H')
    matched_df['match_fill_rate'] = matched_df['matched_quantity'] / matched_df['quantity']

    overall_bucket = matched_df.groupby('time_bucket')['match_fill_rate'].mean().reset_index()
    fig.add_trace(go.Scatter(x=overall_bucket['time_bucket'], y=overall_bucket['match_fill_rate'], name='Overall Match Fill Rate', mode='lines+markers'), row=last_row, col=1)

    for instr in instruments:
        instr_bucket = matched_df[matched_df['instrument'] == instr].groupby('time_bucket')['match_fill_rate'].mean().reset_index()
        fig.add_trace(go.Scatter(x=instr_bucket['time_bucket'], y=instr_bucket['match_fill_rate'], name=f'{instr} Match Fill Rate', mode='lines+markers'), row=last_row, col=1)

    fig.update_yaxes(title_text='Match Fill Rate', row=last_row, col=1)
    fig.update_xaxes(title_text='Time Bucket', row=last_row, col=1)

    fig.update_layout(height=500 * num_rows, width=1500, title_text="Internalization Dashboard with Fill Rate Trend")
    fig.show()

    def plot_internalization_distribution(matched_df):
        matched_df = matched_df.copy()
        matched_df['day'] = matched_df['time'].dt.date
        matched_df['internalization_rate'] = matched_df['matched_quantity'] / matched_df['quantity']

        # Overall daily internalization rate
        daily_overall = matched_df.groupby('day').agg(
            total_qty=('quantity', 'sum'),
            total_matched=('matched_quantity', 'sum')
        )
        daily_overall['daily_internalization_rate'] = daily_overall['total_matched'] / daily_overall['total_qty']

        # By instrument
        daily_by_instr = matched_df.groupby(['instrument', 'day']).agg(
            total_qty=('quantity', 'sum'),
            total_matched=('matched_quantity', 'sum')
        )
        daily_by_instr['daily_internalization_rate'] = daily_by_instr['total_matched'] / daily_by_instr['total_qty']
        daily_by_instr = daily_by_instr.reset_index()

        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=daily_overall['daily_internalization_rate'],
            name='Overall',
            opacity=0.6,
            nbinsx=50
        ))

        for instr in daily_by_instr['instrument'].unique():
            sub = daily_by_instr[daily_by_instr['instrument'] == instr]
            fig.add_trace(go.Histogram(
                x=sub['daily_internalization_rate'],
                name=instr,
                opacity=0.6,
                nbinsx=50
            ))

        fig.update_layout(
            barmode='overlay',
            title='Distribution of Daily Internalization Rate (Overall & by Instrument)',
            xaxis_title='Daily Internalization Rate',
            yaxis_title='Frequency',
            width=1000,
            height=500
        )
        fig.show()


def plot_internalization_distribution(matched_df):
    matched_df = matched_df.copy()
    matched_df['day'] = matched_df['time'].dt.date
    matched_df['internalization_rate'] = matched_df['matched_quantity'] / matched_df['quantity']

    # Overall daily internalization rate
    daily_overall = matched_df.groupby('day').agg(
        total_qty=('quantity', 'sum'),
        total_matched=('matched_quantity', 'sum')
    )
    daily_overall['daily_internalization_rate'] = daily_overall['total_matched'] / daily_overall['total_qty']

    # By instrument
    daily_by_instr = matched_df.groupby(['instrument', 'day']).agg(
        total_qty=('quantity', 'sum'),
        total_matched=('matched_quantity', 'sum')
    )
    daily_by_instr['daily_internalization_rate'] = daily_by_instr['total_matched'] / daily_by_instr['total_qty']
    daily_by_instr = daily_by_instr.reset_index()

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=daily_overall['daily_internalization_rate'],
        name='Overall',
        opacity=0.6,
        nbinsx=50
    ))

    for instr in daily_by_instr['instrument'].unique():
        sub = daily_by_instr[daily_by_instr['instrument'] == instr]
        fig.add_trace(go.Histogram(
            x=sub['daily_internalization_rate'],
            name=instr,
            opacity=0.6,
            nbinsx=50
        ))

    fig.update_layout(
        barmode='overlay',
        title='Distribution of Daily Internalization Rate (Overall & by Instrument)',
        xaxis_title='Daily Internalization Rate',
        yaxis_title='Frequency',
        width=1500,
        height=500
    )
    fig.show()


if __name__ == "__main__":
    df_orders = generate_sample_orders(n=200000, seed=42)
    matched_df = simulate_internalizer_advanced(df_orders, time_window=timedelta(minutes=2))
    matched_df['match_fill_rate'] = matched_df['matched_quantity'] / matched_df['quantity']
    matched_df['day'] = matched_df['time'].dt.date

    overall_rate = compute_internalization_rate(matched_df)
    rate_by_instr = compute_internalization_rate_by_instrument(matched_df)
    full_fill_rate, partial_fill_rate, fill_stats_by_instr = compute_fill_rate_stats(matched_df)

    overall_rate_df = pd.DataFrame([["Overall", overall_rate]], columns=["Type", "Internalization Rate"])
    rate_by_instr_df = rate_by_instr.reset_index()
    fill_stats_df = fill_stats_by_instr.reset_index()

    daily_summary = matched_df.groupby(['instrument', 'day']).agg(
        day_quantity=('quantity', 'sum'),
        day_matched=('matched_quantity', 'sum')
    )

    desc_df = matched_df.groupby(matched_df.time.dt.date).agg(
        day_quantity=('quantity', 'sum'),
        day_matched=('matched_quantity', 'sum'))
    print(desc_df.describe(percentiles=(0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)).round(1))

    print("\n=== Description of Daily Quantity/Matched by Instrument ===")
    for instr in daily_summary.index.get_level_values(0).unique():
        print(f"\n--- Instrument: {instr} ---")
        desc = daily_summary.loc[instr].describe(percentiles=(0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)).round(2)
        print(desc)

    print("\n=== Match Fill Rate per Order (All Instruments) ===")
    print(matched_df['match_fill_rate'].describe(percentiles=(0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)).round(3))

    print("\n=== Match Fill Rate per Order by Instrument ===")
    for instr in matched_df['instrument'].unique():
        subset = matched_df[matched_df['instrument'] == instr]
        print(f"\n--- Instrument: {instr} ---")
        print(subset['match_fill_rate'].describe(percentiles=(0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)).round(3))

    overall_html = overall_rate_df.style.format({"Internalization Rate": "{:.2%}"}).to_html(index=False)
    rate_by_instr_html = rate_by_instr_df.to_html(index=False)
    fill_stats_html = fill_stats_df.style.format({"full_fill_rate": "{:.2%}", "partial_fill_rate": "{:.2%}"}).to_html(index=False)

    combined_html = f"""
           <html>
           <head><title>Internalization Report</title></head>
           <body>
           <h2>Overall Internalization Rate</h2>
           {overall_html}

           <h2>Internalization Rate by Instrument</h2>
           {rate_by_instr_html}

           <h2>Fill Rates by Instrument</h2>
           {fill_stats_html}
           </body>
           </html>
           """

    with open("internalization_report.html", "w") as f:
        f.write(combined_html)
    print("Internalization metrics exported to 'internalization_report.html'")

    plot_combined_dashboard(matched_df)
    plot_internalization_distribution(matched_df)
