def plot_coefficients_by_equation(df, states):
    fig, axes = plt.subplots(4, 2, figsize=(12, 20))

    # The way we defined Z_t implies that the first 5 elements of the
    # state vector correspond to the first variable in y_t, which is GDP growth
    
    title = df.columns
    axes = axes.ravel()
    for i in range(len(title)):
        ax = axes[i]
        #states.iloc[:, i*9+1:i*9+9].plot(ax=ax)
        ax.plot(states.index, states.iloc[:, i*9+1:i*9+9], label=states.iloc[:, i*9+1:i*9+9].columns)
        ax.set_title(title[i])
        ax.legend(loc='upper left',  ncol=1, fontsize=8)  
        ax.set_xlabel(' ')
        ax.set_ylabel('')
        #ax.set_ylim([-1, 3])
    plt.tight_layout()
    plt.savefig('images/TVP-VAR-coefficient.png', dpi=300)
    return ax



# Here, for illustration purposes only, we plot the time-varying
# coefficients conditional on an ad-hoc parameterization

# Recall that `initial_res` contains the Kalman filtering and smoothing,
# and the `states.smoothed` attribute contains the smoothed states

#plot_coefficients_by_equation(df, initial_res.states.filtered);
