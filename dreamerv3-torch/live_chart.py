import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

DEFAULT_BB_CSV = './csv_logs/blackbox_stage2_baseline.csv'
DEFAULT_WB_CSV = './csv_logs/whitebox_stage2_baseline.csv'
REFRESH_MS = 10000


def parse_args():
    p = argparse.ArgumentParser(description='Live training charts for TurtleBot DreamerV3')
    p.add_argument('--run_name', type=str, default=None,
                   help='Derives paths: csv_logs/blackbox_<run_name>.csv and whitebox_<run_name>.csv')
    p.add_argument('--blackbox', type=str, default=None,
                   help='Explicit blackbox CSV path (overrides --run_name)')
    p.add_argument('--whitebox', type=str, default=None,
                   help='Explicit whitebox CSV path (overrides --run_name)')
    p.add_argument('--update_every_episodes', type=int, default=1,
                   help='Redraw only when episode count advances by this many (default: 1)')
    return p.parse_args()


args = parse_args()

if args.blackbox:
    BB_CSV = args.blackbox
elif args.run_name:
    BB_CSV = f'./csv_logs/blackbox_{args.run_name}.csv'
else:
    BB_CSV = DEFAULT_BB_CSV

if args.whitebox:
    WB_CSV = args.whitebox
elif args.run_name:
    WB_CSV = f'./csv_logs/whitebox_{args.run_name}.csv'
else:
    WB_CSV = DEFAULT_WB_CSV

UPDATE_EVERY = max(1, args.update_every_episodes)

print(f'Blackbox CSV : {BB_CSV}')
print(f'Whitebox CSV : {WB_CSV}')
print(f'Redraw every : {UPDATE_EVERY} episodes')

# _last_drawn_episode[0] == -1 means never drawn; this ensures the first call always draws.
_last_drawn_episode = [-1]
# _wb_last_ep starts at -2 (differs from -1) so whitebox draws on the same tick as blackbox's first draw.
_wb_last_ep = [-2]


def load_blackbox():
    if not os.path.exists(BB_CSV):
        return pd.DataFrame()
    try:
        df = pd.read_csv(BB_CSV)
        df['row_index'] = range(len(df))
        return df
    except Exception:
        return pd.DataFrame()


def load_whitebox():
    if not os.path.exists(WB_CSV):
        return pd.DataFrame()
    try:
        return pd.read_csv(WB_CSV)
    except Exception:
        return pd.DataFrame()


fig1, axes1 = plt.subplots(2, 3, figsize=(14, 8))
fig1.suptitle('Black-box Evaluation — Performance Metrics', fontsize=14)


def animate_blackbox(i):
    bb = load_blackbox()
    if bb.empty:
        return

    current_ep = int(bb['episode'].max()) if 'episode' in bb.columns else len(bb)

    # Always draw on first call (_last_drawn_episode == -1); afterwards wait for UPDATE_EVERY new episodes.
    if _last_drawn_episode[0] >= 0 and current_ep < _last_drawn_episode[0] + UPDATE_EVERY:
        return
    _last_drawn_episode[0] = current_ep

    titles = ['Success Rate (%)', 'Collision Rate (%)',
              'Steps to Goal', 'Path Efficiency',
              'Min Obstacle Distance', 'Near Collisions']
    keys   = ['success_rate', 'collision_rate',
              'steps_to_goal', 'path_efficiency',
              'min_obstacle_dist', 'near_collisions']
    colors = ['green', 'red', 'blue', 'orange', 'purple', 'brown']

    for ax, title, key, color in zip(axes1.flatten(), titles, keys, colors):
        ax.cla()
        if key not in bb.columns:
            ax.set_title(title, fontsize=10)
            ax.grid(True, alpha=0.3)
            continue

        data = bb[bb[key] != -1] if key == 'steps_to_goal' else bb
        ax.plot(data['row_index'].values, data[key].values,
                color=color, linewidth=1.5, label='cumulative')

        # Overlay rolling rates on the success and collision panels.
        # dropna() handles older CSV rows that predate the rolling columns.
        if key == 'success_rate':
            for col, style, lbl in [
                ('rolling_success_rate_100', '--', 'rolling 100'),
                ('rolling_success_rate_500', ':',  'rolling 500'),
            ]:
                if col in bb.columns:
                    rs = data[['row_index', col]].dropna()
                    ax.plot(rs['row_index'].values, rs[col].values,
                            color=color, linewidth=1, linestyle=style, alpha=0.7, label=lbl)
            ax.legend(fontsize=7, loc='upper left')

        elif key == 'collision_rate':
            for col, style, lbl in [
                ('rolling_collision_rate_100', '--', 'rolling 100'),
                ('rolling_collision_rate_500', ':',  'rolling 500'),
            ]:
                if col in bb.columns:
                    rc = data[['row_index', col]].dropna()
                    ax.plot(rc['row_index'].values, rc[col].values,
                            color=color, linewidth=1, linestyle=style, alpha=0.7, label=lbl)
            ax.legend(fontsize=7, loc='upper left')

        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Episode')
        ax.grid(True, alpha=0.3)

    fig1.suptitle(f'Black-box — {BB_CSV}  (ep {current_ep})', fontsize=12)
    plt.tight_layout()


fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8))
fig2.suptitle('White-box Evaluation — Internal Model Analysis', fontsize=14)


def animate_whitebox(i):
    # Redraw whitebox only when blackbox has drawn new data (keeps both charts in sync).
    if _last_drawn_episode[0] == _wb_last_ep[0]:
        return
    _wb_last_ep[0] = _last_drawn_episode[0]

    wb = load_whitebox()
    if wb.empty:
        return

    axes2[0, 0].cla()
    if 'train_return' in wb.columns:
        axes2[0, 0].plot(wb['step'].values, wb['train_return'].values,
                         color='blue', linewidth=1.5, label='Return')
    if 'reward_variance' in wb.columns:
        ax_twin = axes2[0, 0].twinx()
        ax_twin.plot(wb['step'].values, wb['reward_variance'].values,
                     color='cyan', linewidth=1, alpha=0.7, label='Variance')
    axes2[0, 0].set_title('Training Convergence')
    axes2[0, 0].set_xlabel('Step')
    axes2[0, 0].grid(True, alpha=0.3)

    axes2[0, 1].cla()
    if 'model_loss' in wb.columns:
        axes2[0, 1].plot(wb['step'].values, wb['model_loss'].values,
                         color='red', linewidth=1.5)
    axes2[0, 1].set_title('Model Loss')
    axes2[0, 1].set_xlabel('Step')
    axes2[0, 1].grid(True, alpha=0.3)

    axes2[1, 0].cla()
    for col, color, lw, alpha, lbl in [
        ('kl',        'green',  1.5, 1.0, 'KL'),
        ('prior_ent', 'orange', 1.0, 0.7, 'Prior Ent'),
        ('post_ent',  'purple', 1.0, 0.7, 'Post Ent'),
    ]:
        if col in wb.columns:
            axes2[1, 0].plot(wb['step'].values, wb[col].values,
                             color=color, linewidth=lw, alpha=alpha, label=lbl)
    axes2[1, 0].set_title('KL Divergence & Entropy')
    axes2[1, 0].set_xlabel('Step')
    axes2[1, 0].legend(fontsize=8)
    axes2[1, 0].grid(True, alpha=0.3)

    axes2[1, 1].cla()
    for col, color, lbl in [('actor_loss', 'blue', 'Actor'), ('value_loss', 'red', 'Value')]:
        if col in wb.columns:
            axes2[1, 1].plot(wb['step'].values, wb[col].values,
                             color=color, linewidth=1.5, label=lbl)
    axes2[1, 1].set_title('Actor & Value Loss')
    axes2[1, 1].set_xlabel('Step')
    axes2[1, 1].legend(fontsize=8)
    axes2[1, 1].grid(True, alpha=0.3)

    fig2.suptitle(f'White-box — {WB_CSV}', fontsize=12)
    plt.tight_layout()


ani1 = animation.FuncAnimation(fig1, animate_blackbox, interval=REFRESH_MS)
ani2 = animation.FuncAnimation(fig2, animate_whitebox, interval=REFRESH_MS)
plt.show()
