import numpy as np
import pandas as pd
import streamlit as st
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, List

# Problem Definition
@dataclass
class GAProblem:
    name: str
    chromosome_type: str
    dim: int
    bounds: Optional[Tuple[float, float]]
    fitness_fn: Callable[[np.ndarray], float]

def make_onemax_target50(dim: int, target: int = 50) -> GAProblem:
    def fitness(x: np.ndarray) -> float:
        count = int(np.sum(x))
        diff = count - target
        return -float(diff * diff)  # max when diff = 0

    return GAProblem(
        name=f"OneMax-Target{target} ({dim} bits)",
        chromosome_type="bit",
        dim=dim,
        bounds=None,
        fitness_fn=fitness,
    )

# Genetic Algorithm
def init_population(problem: GAProblem, pop_size: int, rng: np.random.Generator):
    return rng.integers(0, 2, size=(pop_size, problem.dim), dtype=np.int8)

def tournament_selection(fitness: np.ndarray, k: int, rng: np.random.Generator):
    idx = rng.integers(0, len(fitness), size=k)
    return int(idx[np.argmax(fitness[idx])])

def one_point_crossover(a: np.ndarray, b: np.ndarray, rng: np.random.Generator):
    point = int(rng.integers(1, len(a)))
    return (
        np.concatenate([a[:point], b[point:]]),
        np.concatenate([b[:point], a[point:]])
    )

def bit_mutation(x: np.ndarray, rate: float, rng: np.random.Generator):
    mask = rng.random(x.shape) < rate
    y = x.copy()
    y[mask] = 1 - y[mask]
    return y

def evaluate(pop: np.ndarray, problem: GAProblem):
    return np.array([problem.fitness_fn(ind) for ind in pop], dtype=float)

# GA Running
def run_ga(problem: GAProblem, pop_size, generations, crossover_rate, mutation_rate,
           tournament_k, elitism, seed, live=True):

    rng = np.random.default_rng(seed)
    pop = init_population(problem, pop_size, rng)
    fit = evaluate(pop, problem)

    chart = st.empty()
    text = st.empty()
    progress = st.progress(0)

    history_best = []
    history_avg = []
    history_worst = []

    for gen in range(generations):
        best_idx = int(np.argmax(fit))
        history_best.append(float(fit[best_idx]))
        history_avg.append(float(np.mean(fit)))
        history_worst.append(float(np.min(fit)))

        if live:
            df = pd.DataFrame({
                "Best": history_best,
                "Average": history_avg,
                "Worst": history_worst
            })
            chart.line_chart(df)
            text.write(f"Generation {gen+1}/{generations} | Best fitness: {fit[best_idx]:.6f}")
            progress.progress((gen+1)/generations)

        E = min(elitism, pop_size)
        elite_idx = np.argpartition(fit, -E)[-E:]
        elites = pop[elite_idx]

        new_pop = []
        while len(new_pop) < pop_size - E:
            p1 = pop[tournament_selection(fit, tournament_k, rng)]
            p2 = pop[tournament_selection(fit, tournament_k, rng)]

            if rng.random() < crossover_rate:
                c1, c2 = one_point_crossover(p1, p2, rng)
            else:
                c1, c2 = p1.copy(), p2.copy()

            c1 = bit_mutation(c1, mutation_rate, rng)
            c2 = bit_mutation(c2, mutation_rate, rng)

            new_pop.append(c1)
            if len(new_pop) < pop_size - E:
                new_pop.append(c2)

        pop = np.vstack([np.array(new_pop), elites])
        fit = evaluate(pop, problem)

    final_best_idx = int(np.argmax(fit))
    return pop[final_best_idx], fit[final_best_idx], history_best, history_avg, history_worst

# Streamlit UI
st.set_page_config("Genetic Algorithm", layout="wide")
st.title("Lab Report 2 - Introduction To Artificial Intelligence (Genetic Algorithm)")
st.caption("Name : Aliyah Afifah binti Azril  |  Student ID : SD23006")

st.sidebar.header("GA Parameters")
dim = st.sidebar.number_input("Chromosome length", 1, 5000, 80)
pop_size = st.sidebar.number_input("Population size", 10, 5000, 300)
generations = st.sidebar.number_input("Generations", 1, 5000, 200)
crossover_rate = st.sidebar.slider("Crossover rate", 0.0, 1.0, 0.9)
mutation_rate = st.sidebar.slider("Mutation rate", 0.0, 1.0, 0.01)
tournament_k = st.sidebar.slider("Tournament size", 2, 10, 3)
elitism = st.sidebar.slider("Elitism", 0, 50, 2)
seed = st.sidebar.number_input("Seed", 0, 999999, 42)

problem = make_onemax_target50(dim)

if st.button("Run GA", type="primary"):
    best, best_fit, hb, ha, hw = run_ga(
        problem, pop_size, generations,
        crossover_rate, mutation_rate,
        tournament_k, elitism, seed
    )

    st.subheader("Best Solution")
    st.write(f"Best fitness: {best_fit:.6f}")
    bitstring = ''.join(map(str, best.tolist()))
    st.code(bitstring)
    st.write(f"Number of ones: {np.sum(best)} / {dim}")

    df = pd.DataFrame({"Best": hb, "Average": ha, "Worst": hw})
    st.line_chart(df)
