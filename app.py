from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import os

app = Flask(__name__, static_folder='.')
CORS(app)

GRID = 4
ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
THETA = 1e-4
GOAL = (3, 3)


def get_next_state(i, j, action):
    if action == 'UP':    i = max(i - 1, 0)
    elif action == 'DOWN':  i = min(i + 1, GRID - 1)
    elif action == 'LEFT':  j = max(j - 1, 0)
    elif action == 'RIGHT': j = min(j + 1, GRID - 1)
    return i, j


def get_reward(i, j, goal_reward=1.0, step_reward=-0.04):
    return goal_reward if (i, j) == GOAL else step_reward


def best_action(V, i, j, gamma, step_reward):
    best, bv = None, -float('inf')
    for a in ACTIONS:
        ni, nj = get_next_state(i, j, a)
        q = get_reward(ni, nj, step_reward=step_reward) + gamma * V[ni][nj]
        if q > bv:
            bv, best = q, a
    return best


# ── Policy Iteration ──────────────────────────────────────────────────────────

def policy_evaluation(V, policy, gamma, step_reward):
    """Run full policy evaluation to convergence. Returns updated V and sweep count."""
    sweeps = 0
    while True:
        delta = 0
        new_V = [row[:] for row in V]
        for i in range(GRID):
            for j in range(GRID):
                if (i, j) == GOAL:
                    continue
                ni, nj = get_next_state(i, j, policy[i][j])
                v = get_reward(ni, nj, step_reward=step_reward) + gamma * V[ni][nj]
                delta = max(delta, abs(v - V[i][j]))
                new_V[i][j] = v
        V = new_V
        sweeps += 1
        if delta < THETA:
            break
    return V, sweeps


def policy_improvement(V, policy, gamma, step_reward):
    """Greedy policy improvement. Returns updated policy and stability flag."""
    stable = True
    for i in range(GRID):
        for j in range(GRID):
            if (i, j) == GOAL:
                continue
            old = policy[i][j]
            policy[i][j] = best_action(V, i, j, gamma, step_reward)
            if old != policy[i][j]:
                stable = False
    return policy, stable


def run_policy_iteration(gamma=0.9, step_reward=-0.04):
    V = [[0.0] * GRID for _ in range(GRID)]
    policy = [['RIGHT'] * GRID for _ in range(GRID)]
    history = []
    iteration = 0

    while True:
        iteration += 1
        V, sweeps = policy_evaluation(V, policy, gamma, step_reward)
        policy, stable = policy_improvement(V, policy, gamma, step_reward)

        history.append({
            'iteration': iteration,
            'eval_sweeps': sweeps,
            'stable': stable,
            'V': [row[:] for row in V],
            'policy': [row[:] for row in policy],
        })

        if stable:
            break

    return {
        'algorithm': 'policy_iteration',
        'iterations': iteration,
        'history': history,
        'final_V': V,
        'final_policy': policy,
        'gamma': gamma,
        'step_reward': step_reward,
    }


# ── Value Iteration ───────────────────────────────────────────────────────────

def run_value_iteration(gamma=0.9, step_reward=-0.04):
    V = [[0.0] * GRID for _ in range(GRID)]
    history = []
    iteration = 0

    while True:
        delta = 0
        new_V = [row[:] for row in V]
        iteration += 1

        for i in range(GRID):
            for j in range(GRID):
                if (i, j) == GOAL:
                    continue
                best = max(
                    get_reward(*get_next_state(i, j, a), step_reward=step_reward)
                    + gamma * V[get_next_state(i, j, a)[0]][get_next_state(i, j, a)[1]]
                    for a in ACTIONS
                )
                delta = max(delta, abs(best - V[i][j]))
                new_V[i][j] = best

        V = new_V

        # Extract greedy policy
        policy = [
            [
                'GOAL' if (i, j) == GOAL else best_action(V, i, j, gamma, step_reward)
                for j in range(GRID)
            ]
            for i in range(GRID)
        ]

        history.append({
            'iteration': iteration,
            'delta': delta,
            'V': [row[:] for row in V],
            'policy': [row[:] for row in policy],
        })

        if delta < THETA:
            break

    return {
        'algorithm': 'value_iteration',
        'iterations': iteration,
        'history': history,
        'final_V': V,
        'final_policy': policy,
        'gamma': gamma,
        'step_reward': step_reward,
    }


# ── Step-by-step state (for interactive stepping) ────────────────────────────

_session = {}


def _init_session(algo, gamma, step_reward):
    _session.clear()
    _session.update({
        'algo': algo,
        'gamma': float(gamma),
        'step_reward': float(step_reward),
        'V': [[0.0] * GRID for _ in range(GRID)],
        'policy': [['RIGHT'] * GRID for _ in range(GRID)],
        'pi_phase': 'eval',
        'pi_iter': 0,
        'pi_sub_iter': 0,
        'vi_iter': 0,
        'converged': False,
        'log': [],
    })


def _session_to_response():
    s = _session
    policy = s['policy']
    V = s['V']
    return {
        'V': V,
        'policy': policy,
        'iteration': s['pi_iter'] if s['algo'] == 'pi' else s['vi_iter'],
        'phase': s.get('pi_phase', 'eval'),
        'converged': s['converged'],
        'log': s['log'][-40:],
    }


def _pi_eval_step():
    s = _session
    delta, new_V = 0, [row[:] for row in s['V']]
    for i in range(GRID):
        for j in range(GRID):
            if (i, j) == GOAL:
                continue
            ni, nj = get_next_state(i, j, s['policy'][i][j])
            v = get_reward(ni, nj, step_reward=s['step_reward']) + s['gamma'] * s['V'][ni][nj]
            delta = max(delta, abs(v - s['V'][i][j]))
            new_V[i][j] = v
    s['V'] = new_V
    s['pi_sub_iter'] += 1
    return delta


def _pi_improve_step():
    s = _session
    stable = True
    for i in range(GRID):
        for j in range(GRID):
            if (i, j) == GOAL:
                continue
            old = s['policy'][i][j]
            s['policy'][i][j] = best_action(s['V'], i, j, s['gamma'], s['step_reward'])
            if old != s['policy'][i][j]:
                stable = False
    return stable


def _vi_step():
    s = _session
    delta, new_V = 0, [row[:] for row in s['V']]
    for i in range(GRID):
        for j in range(GRID):
            if (i, j) == GOAL:
                continue
            best = max(
                get_reward(*get_next_state(i, j, a), step_reward=s['step_reward'])
                + s['gamma'] * s['V'][get_next_state(i, j, a)[0]][get_next_state(i, j, a)[1]]
                for a in ACTIONS
            )
            delta = max(delta, abs(best - s['V'][i][j]))
            new_V[i][j] = best
    s['V'] = new_V
    for i in range(GRID):
        for j in range(GRID):
            if (i, j) != GOAL:
                s['policy'][i][j] = best_action(s['V'], i, j, s['gamma'], s['step_reward'])
    s['vi_iter'] += 1
    return delta


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def serve_root():
    return send_from_directory('.', 'index.html')

@app.route('/api/reset', methods=['POST'])
def reset():
    data = request.get_json(silent=True) or request.args or {}
    _init_session(
        algo=data.get('algo', 'pi'),
        gamma=float(data.get('gamma', 0.9)),
        step_reward=float(data.get('step_reward', -0.04)),
    )
    return jsonify({'ok': True, **_session_to_response()})


@app.route('/api/step', methods=['GET', 'POST'])
def step():
    if not _session:
        _init_session('pi', 0.9, -0.04)

    s = _session
    if s['converged']:
        return jsonify({'ok': True, 'already_converged': True, **_session_to_response()})

    delta = None

    if s['algo'] == 'vi':
        delta = _vi_step()
        msg = f"iter {s['vi_iter']:03d}  δ = {delta:.6f}"
        s['log'].append({'type': '', 'msg': msg, 'delta': delta})
        if delta < THETA:
            s['converged'] = True
            s['log'].append({'type': 'ok', 'msg': f"✓ Value Iteration converged in {s['vi_iter']} iterations"})
    else:
        if s['pi_phase'] == 'eval':
            delta = _pi_eval_step()
            s['log'].append({'type': '', 'msg': f"eval  iter {s['pi_iter']:02d} sub {s['pi_sub_iter']:02d}  δ={delta:.5f}", 'delta': delta})
            if delta < THETA:
                s['pi_phase'] = 'improve'
                s['log'].append({'type': 'hi', 'msg': '→ evaluation converged, improving policy…'})
        else:
            stable = _pi_improve_step()
            s['pi_iter'] += 1
            s['pi_sub_iter'] = 0
            s['log'].append({'type': '', 'msg': f"impr  iter {s['pi_iter']:03d}  stable={stable}"})
            if stable:
                s['converged'] = True
                s['log'].append({'type': 'ok', 'msg': f"✓ Policy Iteration converged in {s['pi_iter']} iterations"})
            else:
                s['pi_phase'] = 'eval'

    return jsonify({'ok': True, 'delta': delta, **_session_to_response()})


@app.route('/api/run', methods=['GET', 'POST'])
def run_full():
    """Run a complete algorithm and return the full result at once."""
    data = request.get_json(silent=True) or request.args or {}
    algo = data.get('algo', 'pi')
    gamma = float(data.get('gamma', 0.9))
    step_reward = float(data.get('step_reward', -0.04))

    if algo == 'vi':
        result = run_value_iteration(gamma=gamma, step_reward=step_reward)
    else:
        result = run_policy_iteration(gamma=gamma, step_reward=step_reward)

    return jsonify(result)


@app.route('/api/compare', methods=['GET', 'POST'])
def compare():
    """Run both algorithms and return side-by-side results."""
    data = request.get_json(silent=True) or request.args or {}
    gamma = float(data.get('gamma', 0.9))
    step_reward = float(data.get('step_reward', -0.04))

    pi = run_policy_iteration(gamma=gamma, step_reward=step_reward)
    vi = run_value_iteration(gamma=gamma, step_reward=step_reward)

    return jsonify({
        'policy_iteration': {
            'iterations': pi['iterations'],
            'total_eval_sweeps': sum(h['eval_sweeps'] for h in pi['history']),
            'final_V': pi['final_V'],
            'final_policy': pi['final_policy'],
        },
        'value_iteration': {
            'iterations': vi['iterations'],
            'final_delta': vi['history'][-1]['delta'],
            'final_V': vi['final_V'],
            'final_policy': vi['final_policy'],
        },
        'gamma': gamma,
        'step_reward': step_reward,
    })


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'grid': GRID, 'goal': list(GOAL)})


if __name__ == '__main__':
    print("Grid World RL API running at http://localhost:5000")
    print("Endpoints: /api/reset  /api/step  /api/run  /api/compare  /api/health")
    app.run(debug=True, port=5000)
