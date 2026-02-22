# ---- Cleaner Monte Carlo Visualization ----

st.subheader("🎲 Monte Carlo Simulation (Clean View)")

final_prices = simulations

mean_path = np.mean(final_prices, axis=1)
upper_band = np.percentile(final_prices, 95, axis=1)
lower_band = np.percentile(final_prices, 5, axis=1)

mc_fig = go.Figure()

# Mean path
mc_fig.add_trace(go.Scatter(
    y=mean_path,
    mode='lines',
    name='Expected Path',
    line=dict(width=3)
))

# Upper band
mc_fig.add_trace(go.Scatter(
    y=upper_band,
    mode='lines',
    name='95% Upper Bound',
    line=dict(dash='dash')
))

# Lower band
mc_fig.add_trace(go.Scatter(
    y=lower_band,
    mode='lines',
    name='5% Lower Bound',
    line=dict(dash='dash')
))

mc_fig.update_layout(template="plotly_dark")

st.plotly_chart(mc_fig, use_container_width=True)
