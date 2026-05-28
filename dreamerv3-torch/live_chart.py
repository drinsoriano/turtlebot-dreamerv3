import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

BB_CSV = './csv_logs/blackbox_stage2_baseline.csv'
WB_CSV = './csv_logs/whitebox_stage2_baseline.csv'
REFRESH_MS = 10000

def load_blackbox():
    if not os.path.exists(BB_CSV):
        return pd.DataFrame()
    try:
        df = pd.read_csv(BB_CSV)
        df['row_index'] = range(len(df))
        return df
    except:
        return pd.DataFrame()

def load_whitebox():
    if not os.path.exists(WB_CSV):
        return pd.DataFrame()
    try:
        return pd.read_csv(WB_CSV)
    except:
        return pd.DataFrame()

fig1, axes1 = plt.subplots(2, 3, figsize=(14, 8))
fig1.suptitle('Black-box Evaluation — Performance Metrics', fontsize=14)

def animate_blackbox(i):
    bb = load_blackbox()
    if bb.empty:
        return
    titles = ['Success Rate (%)', 'Collision Rate (%)',
              'Steps to Goal', 'Path Efficiency',
              'Min Obstacle Distance', 'Near Collisions']
    keys   = ['success_rate', 'collision_rate',
              'steps_to_goal', 'path_efficiency',
              'min_obstacle_dist', 'near_collisions']
    colors = ['green', 'red', 'blue', 'orange', 'purple', 'brown']
    for ax, title, key, color in zip(axes1.flatten(), titles, keys, colors):
        ax.cla()
        if key in bb.columns:
            data = bb[bb[key] != -1] if key == 'steps_to_goal' else bb
            ax.plot(data['row_index'].values, data[key].values, color=color, linewidth=1.5)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Episode (total)')
        ax.grid(True, alpha=0.3)
    plt.tight_layout()

fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8))
fig2.suptitle('White-box Evaluation — Internal Model Analysis', fontsize=14)

def animate_whitebox(i):
    wb = load_whitebox()
    if wb.empty:
        return
    axes2[0,0].cla()
    if 'train_return' in wb.columns:
        axes2[0,0].plot(wb['step'].values, wb['train_return'].values, color='blue', linewidth=1.5, label='Return')
    if 'reward_variance' in wb.columns:
        ax2 = axes2[0,0].twinx()
        ax2.plot(wb['step'].values, wb['reward_variance'].values, color='cyan', linewidth=1, alpha=0.7, label='Variance')
    axes2[0,0].set_title('Training Convergence')
    axes2[0,0].set_xlabel('Step')
    axes2[0,0].grid(True, alpha=0.3)

    axes2[0,1].cla()
    if 'model_loss' in wb.columns:
        axes2[0,1].plot(wb['step'].values, wb['model_loss'].values, color='red', linewidth=1.5)
    axes2[0,1].set_title('Model Loss')
    axes2[0,1].set_xlabel('Step')
    axes2[0,1].grid(True, alpha=0.3)

    axes2[1,0].cla()
    if 'kl' in wb.columns:
        axes2[1,0].plot(wb['step'].values, wb['kl'].values, color='green', linewidth=1.5, label='KL')
    if 'prior_ent' in wb.columns:
        axes2[1,0].plot(wb['step'].values, wb['prior_ent'].values, color='orange', linewidth=1, alpha=0.7, label='Prior Ent')
    if 'post_ent' in wb.columns:
        axes2[1,0].plot(wb['step'].values, wb['post_ent'].values, color='purple', linewidth=1, alpha=0.7, label='Post Ent')
    axes2[1,0].set_title('KL Divergence & Entropy')
    axes2[1,0].set_xlabel('Step')
    axes2[1,0].legend(fontsize=8)
    axes2[1,0].grid(True, alpha=0.3)

    axes2[1,1].cla()
    if 'actor_loss' in wb.columns:
        axes2[1,1].plot(wb['step'].values, wb['actor_loss'].values, color='blue', linewidth=1.5, label='Actor')
    if 'value_loss' in wb.columns:
        axes2[1,1].plot(wb['step'].values, wb['value_loss'].values, color='red', linewidth=1.5, label='Value')
    axes2[1,1].set_title('Actor & Value Loss')
    axes2[1,1].set_xlabel('Step')
    axes2[1,1].legend(fontsize=8)
    axes2[1,1].grid(True, alpha=0.3)

    plt.tight_layout()

ani1 = animation.FuncAnimation(fig1, animate_blackbox, interval=REFRESH_MS)
ani2 = animation.FuncAnimation(fig2, animate_whitebox, interval=REFRESH_MS)
plt.show()