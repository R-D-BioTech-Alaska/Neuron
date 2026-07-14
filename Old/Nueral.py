
import sys
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # Suppress TensorFlow INFO and WARNING messages

import json
import time
import logging
import traceback
import threading
import multiprocessing
import concurrent.futures
from collections import defaultdict
from typing import List, Dict, Tuple
import queue

import numpy as np
import nltk
from nltk.tokenize import word_tokenize

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.circuit import Parameter
from qiskit_aer.noise import NoiseModel, depolarizing_error  # Added Import

import tensorflow as tf  # TensorFlow import after setting environment variables
from tensorflow import keras
from tensorflow.keras.utils import plot_model

try:
    import psutil
except ImportError:
    psutil = None

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

nltk.download('punkt', quiet=True)

def normalize_vector(vec: np.ndarray) -> np.ndarray:

    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        return vec.copy()
    return vec / norm

# ============================
# Quantum Neuron (20-Qubit Logical Unit) *If changing, stay below 100 or degredation occurs. *Outdated*
# ============================

class QuantumNeuron:

    def __init__(self, qubits: List[int]):

        if len(qubits) != 20:
            sys.exit("Error: QuantumNeuron requires exactly 20 qubits.")
        self.qubits = qubits
        self.circuit = QuantumCircuit(20)
        self._encode_logical_qubit()
        self._apply_processing()

    def _encode_logical_qubit(self):

        for i in range(10):
            self.circuit.h(i)
        
        for i in range(10):
            self.circuit.cx(i, i + 10)
        
        for i in range(10):
            self.circuit.cz(i, i + 10)
            self.circuit.cx(i + 10, i)
        
        self.circuit.barrier()

    def _apply_processing(self):

        for i in range(20):
            self.circuit.rx(0.5, i)
        
        self.circuit.barrier()

    def get_circuit(self) -> QuantumCircuit:

        return self.circuit.copy()

class GrapheneLayer:

    def __init__(self, num_neurons: int, starting_qubit: int):

        self.num_neurons = num_neurons
        self.starting_qubit = starting_qubit
        self.qubit_mapping = self._assign_qubits()
        self.circuit = QuantumCircuit(self.num_neurons * 20)
        self._build_layer()

    def _assign_qubits(self) -> List[List[int]]:

        mapping = []
        for n in range(self.num_neurons):
            qubits = list(range(self.starting_qubit + n * 20, self.starting_qubit + (n + 1) * 20))
            mapping.append(qubits)
        return mapping

    def _build_layer(self):

        for neuron_qubits in self.qubit_mapping:
            neuron = QuantumNeuron(neuron_qubits)
            self.circuit.compose(neuron.get_circuit(), qubits=list(range(len(neuron_qubits))), inplace=True)

        for i in range(self.num_neurons - 1):
            last_qubit_current = (i + 1) * 20 - 1
            first_qubit_next = (i + 1) * 20
            self.circuit.cz(last_qubit_current, first_qubit_next)
        
        self.circuit.barrier()

    def get_circuit(self) -> QuantumCircuit:

        return self.circuit.copy()

class StackedLayers:

    def __init__(self, num_layers: int, neurons_per_layer: int):

        self.num_layers = num_layers
        self.neurons_per_layer = neurons_per_layer
        self.circuit = QuantumCircuit(num_layers * neurons_per_layer * 20)
        self._build_stacked_layers()

    def _build_stacked_layers(self):

        for layer_num in range(self.num_layers):
            starting_qubit = layer_num * self.neurons_per_layer * 20
            graphene_layer = GrapheneLayer(self.neurons_per_layer, starting_qubit)
            self.circuit.compose(
                graphene_layer.get_circuit(),
                qubits=list(range(starting_qubit, starting_qubit + self.neurons_per_layer * 20)),
                inplace=True
            )
            
            if layer_num > 0:
                prev_layer_last_qubit = (layer_num - 1) * self.neurons_per_layer * 20 + self.neurons_per_layer * 20 - 1
                current_layer_first_qubit = layer_num * self.neurons_per_layer * 20
                self.circuit.cz(prev_layer_last_qubit, current_layer_first_qubit)
        
        self.circuit.barrier()

    def get_circuit(self) -> QuantumCircuit:

        return self.circuit.copy()


class AdamOptimizer:

    def __init__(self, parameters: np.ndarray, lr: float = 0.001, betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8):
        self.parameters = parameters
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.m = np.zeros_like(self.parameters)
        self.v = np.zeros_like(self.parameters)
        self.t = 0

    def step(self, gradients: np.ndarray):
        self.t += 1
        self.m = self.betas[0] * self.m + (1 - self.betas[0]) * gradients
        self.v = self.betas[1] * self.v + (1 - self.betas[1]) * (gradients ** 2)
        m_hat = self.m / (1 - self.betas[0] ** self.t)
        v_hat = self.v / (1 - self.betas[1] ** self.t)
        update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        self.parameters -= update
        return self.parameters

class QuantumLanguageModel:

    def __init__(self, vocab_size: int, embed_dim: int, num_layers: int, neurons_per_layer: int,
                 sim_method: str = 'cpu', num_threads: int = 1, enable_logging: bool = True):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.neurons_per_layer = neurons_per_layer

        self.embeddings = (np.random.randn(vocab_size, embed_dim) * 0.01).astype(np.float32)

        self.stacked_layers = StackedLayers(num_layers, neurons_per_layer)

        self.W_proj = (np.random.randn(embed_dim, embed_dim) * 0.01).astype(np.float32)  # Example projection matrix
        self.W_out = (np.random.randn(vocab_size, embed_dim) * 0.01).astype(np.float32)  # Output matrix

        self._initialize_quantum_params()

        self.sim_method = sim_method
        self.num_threads = num_threads
        self.backend = self.initialize_simulator()

    def _initialize_quantum_params(self):

        self.rotation_angles = {}
        total_qubits = self.stacked_layers.circuit.num_qubits
        for qubit in range(total_qubits):
            self.rotation_angles[qubit] = np.random.uniform(0, 2 * np.pi)

    def initialize_simulator(self):

        if self.sim_method == 'gpu':
            try:
                backend = AerSimulator(method='statevector', device='GPU', max_parallel_threads=self.num_threads)
                logging.info(f"QuantumLanguageModel: Using GPU for simulation.")
            except Exception as e:
                logging.warning(f"QuantumLanguageModel GPU initialization error: {e}. Falling back to CPU.")
                backend = AerSimulator(method='statevector', max_parallel_threads=self.num_threads)
                logging.info(f"QuantumLanguageModel: Using CPU for simulation.")
        else:
            backend = AerSimulator(method='statevector', max_parallel_threads=self.num_threads)
            logging.info(f"QuantumLanguageModel: Using CPU for simulation.")
        return backend

    def encode_input(self, input_id: int) -> np.ndarray:

        if input_id >= self.vocab_size:
            raise ValueError(f"Input ID {input_id} exceeds vocabulary size {self.vocab_size}.")
        return self.embeddings[input_id]

    def build_full_circuit(self, input_id: int) -> QuantumCircuit:

        input_embedding = self.encode_input(input_id)
        input_circuit = QuantumCircuit(self.stacked_layers.circuit.num_qubits)

        num_qubits = self.stacked_layers.circuit.num_qubits
        state_prep_length = 2**num_qubits
        if len(input_embedding) > state_prep_length:
            raise ValueError("Embedding dimension exceeds statevector size.")

        state_prep_vec = np.zeros(state_prep_length, dtype=complex)
        state_prep_vec[:len(input_embedding)] = input_embedding.astype(complex)
        state_prep_vec = normalize_vector(state_prep_vec)
        input_circuit.initialize(state_prep_vec, qubits=range(num_qubits))

        full_circuit = QuantumCircuit(num_qubits, self.vocab_size)
        full_circuit.compose(input_circuit, inplace=True)
        full_circuit.compose(self.stacked_layers.get_circuit(), inplace=True)

        for qubit in range(num_qubits):
            angle = self.rotation_angles.get(qubit, 0.0)
            full_circuit.rx(angle, qubit)

        full_circuit.barrier()

        for token in range(self.vocab_size):
            if token < num_qubits:
                full_circuit.measure(token, token)
            else:
                # For tokens beyond the number of qubits, map to the closest qubit *This shouldn't happen, but incase it does then this will need to be altered*
                mapped_qubit = token % num_qubits
                full_circuit.measure(mapped_qubit, token)

        return full_circuit

    def forward(self, input_id: int) -> np.ndarray:

        circuit = self.build_full_circuit(input_id)
        job = self.backend.run(circuit, shots=1024)
        result = job.result()
        counts = result.get_counts()

        logits = np.zeros(self.vocab_size, dtype=np.float32)
        for outcome, count in counts.items():

            bits = outcome[::-1]
            for idx, bit in enumerate(bits):
                if idx < self.vocab_size:
                    logits[idx] += count if bit == '1' else 0
        return logits

    def _get_noise_model(self) -> NoiseModel:

        noise_model = NoiseModel()
        depol_error = depolarizing_error(0.01, 1)  # 1% error rate for single-qubit gates
        depol_cx_error = depolarizing_error(0.02, 2)  # 2% error rate for two-qubit gates
        noise_model.add_all_qubit_quantum_error(depol_error, ['rx', 'ry', 'rz', 'h', 'cx', 'cz'])
        noise_model.add_all_qubit_quantum_error(depol_cx_error, ['cx', 'cz'])
        return noise_model

    def simulate_and_evaluate(self, input_id: int, target_id: int) -> float:

        circuit = self.build_full_circuit(input_id)
        job = self.backend.run(circuit, noise_model=self._get_noise_model(), shots=1024)
        result = job.result()
        counts = result.get_counts()

        ideal_circuit = QuantumCircuit(circuit.num_qubits, self.vocab_size)
        input_embedding = self.encode_input(input_id)
        normalized_embedding = normalize_vector(input_embedding)
        state_prep_length = 2**self.stacked_layers.circuit.num_qubits
        if len(normalized_embedding) > state_prep_length:
            raise ValueError("Embedding dimension exceeds statevector size.")
        state_prep_vec = np.zeros(state_prep_length, dtype=complex)
        state_prep_vec[:len(normalized_embedding)] = normalized_embedding.astype(complex)
        state_prep_vec = normalize_vector(state_prep_vec)
        ideal_circuit.initialize(state_prep_vec, qubits=range(self.stacked_layers.circuit.num_qubits))
        ideal_circuit.compose(self.stacked_layers.get_circuit(), inplace=True)

        for qubit in range(self.stacked_layers.circuit.num_qubits):
            ideal_circuit.rx(0.0, qubit)

        ideal_circuit.barrier()

        for token in range(self.vocab_size):
            if token < self.stacked_layers.circuit.num_qubits:
                ideal_circuit.measure(token, token)
            else:

                mapped_qubit = token % self.stacked_layers.circuit.num_qubits
                ideal_circuit.measure(mapped_qubit, token)

        ideal_job = self.backend.run(ideal_circuit, shots=1024)
        ideal_result = ideal_job.result()
        ideal_counts = ideal_result.get_counts()
      
        fidelity_scores = []
        for token in range(self.vocab_size):
            actual = counts.get(f"{token:0{self.stacked_layers.circuit.num_qubits}b}", 0) / 1024
            ideal = ideal_counts.get(f"{token:0{self.stacked_layers.circuit.num_qubits}b}", 0) / 1024
            fidelity_scores.append(state_fidelity(actual, ideal))

        avg_fidelity = np.mean(fidelity_scores)
        logging.info(f"Fidelity for input ID {input_id} targeting ID {target_id}: {avg_fidelity:.4f}")
        return avg_fidelity

    def _get_all_parameters(self) -> np.ndarray:

        return np.array(list(self.rotation_angles.values()))

    def set_all_parameters(self, params: np.ndarray):

        if len(params) != len(self.rotation_angles):
            raise ValueError("Parameter array length mismatch.")
        for idx, qubit in enumerate(self.rotation_angles.keys()):
            self.rotation_angles[qubit] = params[idx]

    def to_dict(self) -> dict:

        return {
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "num_layers": self.num_layers,
            "neurons_per_layer": self.neurons_per_layer,
            "embeddings": self.embeddings.tolist(),
            "W_proj": self.W_proj.tolist(),
            "W_out": self.W_out.tolist(),
            "rotation_angles": self.rotation_angles,
            "version": "2.0"  # Updated version
        }

    def from_dict(self, d: dict):

        if d.get("version") != "2.0":
            raise ValueError("Unsupported model version.")
        self.vocab_size = d["vocab_size"]
        self.embed_dim = d["embed_dim"]
        self.num_layers = d["num_layers"]
        self.neurons_per_layer = d["neurons_per_layer"]
        self.embeddings = np.array(d["embeddings"], dtype=np.float32)
        self.W_proj = np.array(d["W_proj"], dtype=np.float32)
        self.W_out = np.array(d["W_out"], dtype=np.float32)
        self.rotation_angles = d["rotation_angles"]

    def save_model(self, save_path: str):

        model_dict = self.to_dict()
        try:
            with open(save_path, 'w') as f:
                json.dump(model_dict, f)
            logging.info(f"Model saved to {save_path}")
        except Exception as e:
            logging.error(f"Failed to save model: {e}")
            raise

    def load_model(self, load_path: str):

        if not os.path.exists(load_path):
            raise FileNotFoundError(f"File {load_path} does not exist.")
        try:
            with open(load_path, 'r') as f:
                model_dict = json.load(f)
            self.from_dict(model_dict)
            logging.info(f"Model loaded from {load_path}")
            self.backend = self.initialize_simulator()
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise

def compute_gradient_for_parameter(args):

    (vocab_size, embed_dim, num_layers, neurons_per_layer, sim_method, num_threads, X, Y, original_params, i) = args
    try:
        model = QuantumLanguageModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            neurons_per_layer=neurons_per_layer,
            sim_method=sim_method,
            num_threads=num_threads,
            enable_logging=False  # Suppress logging during gradient computation
        )
        model.set_all_parameters(original_params)

        shift = np.pi / 2
        model.rotation_angles[i] += shift
        loss_plus = np.mean([
            cross_entropy_loss(model.forward(x), y)
            for x, y in zip(X, Y)
        ])

        model.rotation_angles[i] -= 2 * shift
        loss_minus = np.mean([
            cross_entropy_loss(model.forward(x), y)
            for x, y in zip(X, Y)
        ])

        model.rotation_angles[i] += shift

        gradient = (loss_plus - loss_minus) / 2.0
        return i, gradient
    except Exception:
        traceback.print_exc()
        return i, 0.0

def compute_gradients_parallel(model: QuantumLanguageModel, X: np.ndarray, Y: np.ndarray, num_processes: int = 1, progress_callback=None) -> np.ndarray:

    gradients = np.zeros_like(model._get_all_parameters())
    original_params = model._get_all_parameters().copy()
    total_params = len(original_params)

    block_size = 10000  # Adjust as needed
    blocks = (total_params + block_size - 1) // block_size

    args_list = [
        (
            model.vocab_size,
            model.embed_dim,
            model.num_layers,
            model.neurons_per_layer,
            model.sim_method,
            model.num_threads,
            X,
            Y,
            original_params,
            i
        )
        for i in range(total_params)
    ]

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = {executor.submit(compute_gradient_for_parameter, args): args[-1] for args in args_list}
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            i, gradient = future.result()
            gradients[i] = gradient
            completed += 1
            if completed % block_size == 0 or completed == total_params:
                if progress_callback:
                    progress_callback(completed, total_params, i, gradient)

    return gradients

def cross_entropy_loss(logits: np.ndarray, target: np.ndarray) -> float:

    logits = logits - np.max(logits)
    softmax = np.exp(logits) / np.sum(np.exp(logits))
    softmax = np.clip(softmax, 1e-12, 1.0)
    return -np.sum(target * np.log(softmax))

def perplexity(logits: np.ndarray, target: int) -> float:

    ce_loss = cross_entropy_loss(logits, np.eye(len(logits))[target])
    return np.exp(ce_loss)

def bleu_score(reference: List[str], hypothesis: List[str], max_n: int = 4) -> float:

    from collections import Counter
    import math

    weights = [1.0 / max_n] * max_n
    reference_counts = [Counter([tuple(reference[i:i+n]) for i in range(len(reference)-n+1)]) for n in range(1, max_n+1)]
    hypothesis_counts = [Counter([tuple(hypothesis[i:i+n]) for i in range(len(hypothesis)-n+1)]) for n in range(1, max_n+1)]

    precisions = []
    for ref_count, hyp_count in zip(reference_counts, hypothesis_counts):
        overlap = hyp_count & ref_count
        precision = sum(overlap.values()) / max(sum(hyp_count.values()), 1e-12)
        precisions.append(precision)

    ref_length = len(reference)
    hyp_length = len(hypothesis)
    if hyp_length == 0:
        bp = 0
    elif hyp_length > ref_length:
        bp = 1
    else:
        bp = math.exp(1 - ref_length / hyp_length)

    if min(precisions) > 0:
        log_precisions = [w * math.log(p) for w, p in zip(weights, precisions)]
        geo_mean = math.exp(sum(log_precisions))
    else:
        geo_mean = 0

    bleu = bp * geo_mean
    return bleu

def quantum_loss(params: np.ndarray, model: QuantumLanguageModel, X: np.ndarray, Y: np.ndarray) -> float:
    model.set_all_parameters(params)
    
    total_loss = 0.0
    for input_id, target in zip(X, Y):
        logits = model.forward(input_id)
        loss = cross_entropy_loss(logits, target)
        total_loss += loss
    
    avg_loss = total_loss / len(X)
    return avg_loss

def train_quantum_model(model: QuantumLanguageModel, X: np.ndarray, Y: np.ndarray, epochs: int = 10, learning_rate: float = 0.001, num_threads: int = 1, log_queue: queue.Queue = None, stop_flag=None, time_lock: threading.Lock = None, time_data=None, optimizer=None):

    if time_data is None:
        time_data = {}
    start_time = time_data['start_time'] = time.time()
    time_data['epochs_done'] = 0
    time_data['epochs'] = epochs

    for epoch in range(epochs):
        if stop_flag and stop_flag.is_set():
            if log_queue:
                log_queue.put("Training stopped by user.\n")
            break

        if log_queue:
            log_queue.put(f"Starting Epoch {epoch+1}/{epochs}\n")

        epoch_start_time = time.time()

        def progress_callback(completed, total, param_index, gradient):
            progress = (completed / total) * 100
            if log_queue:
                log_queue.put(f"Gradient Computation Progress: {completed}/{total} parameters ({progress:.2f}%) completed.\n")

        gradients = compute_gradients_parallel(model, X, Y, num_processes=num_threads, progress_callback=progress_callback)

        if optimizer:
            updated_params = optimizer.step(gradients)
            model.set_all_parameters(updated_params)
        else:
            params = model._get_all_parameters()
            params -= learning_rate * gradients
            model.set_all_parameters(params)

        total_loss = np.mean([
            cross_entropy_loss(model.forward(x), y)
            for x, y in zip(X, Y)
        ])
        total_perplexity = np.mean([
            perplexity(model.forward(x), np.argmax(y))
            for x, y in zip(X, Y)
        ])

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        if log_queue:
            log_queue.put(f"Epoch {epoch+1}/{epochs} completed in {epoch_duration:.2f}s, Average Loss: {total_loss:.6f}, Perplexity: {total_perplexity:.6f}\n")

        if time_lock:
            with time_lock:
                time_data['epochs_done'] = epoch + 1
                elapsed = epoch_end_time - start_time
                if time_data['epochs_done'] > 0 and time_data['epochs_done'] < epochs:
                    per_epoch = elapsed / time_data['epochs_done']
                    remaining = (epochs - time_data['epochs_done']) * per_epoch
                    time_data['remaining'] = remaining
                else:
                    time_data['remaining'] = 0

        epoch_progress = ((epoch + 1) / epochs) * 100
        if epoch_progress > 100:
            epoch_progress = 100
        if log_queue:
            log_queue.put(f"Epoch Progress: {epoch_progress:.2f}%\n")

    if log_queue and (not stop_flag or not stop_flag.is_set()):
        log_queue.put("Training completed.\n")

def softmax(x: np.ndarray) -> np.ndarray:

    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def run_inference(model: QuantumLanguageModel, input_sequence: List[int], token_to_id: Dict[str, int], id_to_token: Dict[int, str], max_length: int = 50, temperature: float = 1.0, log_callback=None):

    generated = input_sequence.copy()
    for _ in range(max_length):
        logits = model.forward(generated[-1])
        probabilities = softmax(logits / temperature)

        chosen_index = np.random.choice(len(probabilities), p=probabilities)
        generated.append(chosen_index)

        if chosen_index == token_to_id.get("<END>", chosen_index):
            break

    generated_tokens = [id_to_token.get(idx, "<UNK>") for idx in generated]
    response = " ".join(generated_tokens)

    if log_callback:
        log_callback(f"Generated Response:\n{response}\n\n")

    return generated_tokens, response

def quantum_loss(params: np.ndarray, model: QuantumLanguageModel, X: np.ndarray, Y: np.ndarray) -> float:

    model.set_all_parameters(params)
    
    total_loss = 0.0
    for input_id, target in zip(X, Y):
        logits = model.forward(input_id)
        loss = cross_entropy_loss(logits, target)
        total_loss += loss
    
    avg_loss = total_loss / len(X)
    return avg_loss

def train_quantum_model(model: QuantumLanguageModel, X: np.ndarray, Y: np.ndarray, epochs: int = 10, learning_rate: float = 0.001, num_threads: int = 1, log_queue: queue.Queue = None, stop_flag=None, time_lock: threading.Lock = None, time_data=None, optimizer=None):

    if time_data is None:
        time_data = {}
    start_time = time_data['start_time'] = time.time()
    time_data['epochs_done'] = 0
    time_data['epochs'] = epochs

    for epoch in range(epochs):
        if stop_flag and stop_flag.is_set():
            if log_queue:
                log_queue.put("Training stopped by user.\n")
            break

        if log_queue:
            log_queue.put(f"Starting Epoch {epoch+1}/{epochs}\n")

        epoch_start_time = time.time()

        def progress_callback(completed, total, param_index, gradient):

            progress = (completed / total) * 100
            if log_queue:
                log_queue.put(f"Gradient Computation Progress: {completed}/{total} parameters ({progress:.2f}%) completed.\n")

        gradients = compute_gradients_parallel(model, X, Y, num_processes=num_threads, progress_callback=progress_callback)

        if optimizer:
            updated_params = optimizer.step(gradients)
            model.set_all_parameters(updated_params)
        else:
            params = model._get_all_parameters()
            params -= learning_rate * gradients
            model.set_all_parameters(params)

        total_loss = np.mean([
            cross_entropy_loss(model.forward(x), y)
            for x, y in zip(X, Y)
        ])
        total_perplexity = np.mean([
            perplexity(model.forward(x), np.argmax(y))
            for x, y in zip(X, Y)
        ])

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        if log_queue:
            log_queue.put(f"Epoch {epoch+1}/{epochs} completed in {epoch_duration:.2f}s, Average Loss: {total_loss:.6f}, Perplexity: {total_perplexity:.6f}\n")


        if time_lock:
            with time_lock:
                time_data['epochs_done'] = epoch + 1
                elapsed = epoch_end_time - start_time
                if time_data['epochs_done'] > 0 and time_data['epochs_done'] < epochs:
                    per_epoch = elapsed / time_data['epochs_done']
                    remaining = (epochs - time_data['epochs_done']) * per_epoch
                    time_data['remaining'] = remaining
                else:
                    time_data['remaining'] = 0

        epoch_progress = ((epoch + 1) / epochs) * 100
        if epoch_progress > 100:
            epoch_progress = 100
        if log_queue:
            log_queue.put(f"Epoch Progress: {epoch_progress:.2f}%\n")

    if log_queue and (not stop_flag or not stop_flag.is_set()):
        log_queue.put("Training completed.\n")

def softmax(x: np.ndarray) -> np.ndarray:

    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def run_inference(model: QuantumLanguageModel, input_sequence: List[int], token_to_id: Dict[str, int], id_to_token: Dict[int, str], max_length: int = 50, temperature: float = 1.0, log_callback=None):
    generated = input_sequence.copy()
    for _ in range(max_length):
        logits = model.forward(generated[-1])
        probabilities = softmax(logits / temperature)

        chosen_index = np.random.choice(len(probabilities), p=probabilities)
        generated.append(chosen_index)

        if chosen_index == token_to_id.get("<END>", chosen_index):
            break

    generated_tokens = [id_to_token.get(idx, "<UNK>") for idx in generated]
    response = " ".join(generated_tokens)

    if log_callback:
        log_callback(f"Generated Response:\n{response}\n\n")

    return generated_tokens, response

def quantum_loss(params: np.ndarray, model: QuantumLanguageModel, X: np.ndarray, Y: np.ndarray) -> float:

    model.set_all_parameters(params)
    
    total_loss = 0.0
    for input_id, target in zip(X, Y):
        logits = model.forward(input_id)
        loss = cross_entropy_loss(logits, target)
        total_loss += loss
    
    avg_loss = total_loss / len(X)
    return avg_loss

def train_quantum_model(model: QuantumLanguageModel, X: np.ndarray, Y: np.ndarray, epochs: int = 10, learning_rate: float = 0.001, num_threads: int = 1, log_queue: queue.Queue = None, stop_flag=None, time_lock: threading.Lock = None, time_data=None, optimizer=None):

    if time_data is None:
        time_data = {}
    start_time = time_data['start_time'] = time.time()
    time_data['epochs_done'] = 0
    time_data['epochs'] = epochs

    for epoch in range(epochs):
        if stop_flag and stop_flag.is_set():
            if log_queue:
                log_queue.put("Training stopped by user.\n")
            break

        if log_queue:
            log_queue.put(f"Starting Epoch {epoch+1}/{epochs}\n")

        epoch_start_time = time.time()

        def progress_callback(completed, total, param_index, gradient):

            progress = (completed / total) * 100
            if log_queue:
                log_queue.put(f"Gradient Computation Progress: {completed}/{total} parameters ({progress:.2f}%) completed.\n")

        gradients = compute_gradients_parallel(model, X, Y, num_processes=num_threads, progress_callback=progress_callback)

        if optimizer:
            updated_params = optimizer.step(gradients)
            model.set_all_parameters(updated_params)
        else:
            params = model._get_all_parameters()
            params -= learning_rate * gradients
            model.set_all_parameters(params)

        total_loss = np.mean([
            cross_entropy_loss(model.forward(x), y)
            for x, y in zip(X, Y)
        ])
        total_perplexity = np.mean([
            perplexity(model.forward(x), np.argmax(y))
            for x, y in zip(X, Y)
        ])

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        if log_queue:
            log_queue.put(f"Epoch {epoch+1}/{epochs} completed in {epoch_duration:.2f}s, Average Loss: {total_loss:.6f}, Perplexity: {total_perplexity:.6f}\n")

        if time_lock:
            with time_lock:
                time_data['epochs_done'] = epoch + 1
                elapsed = epoch_end_time - start_time
                if time_data['epochs_done'] > 0 and time_data['epochs_done'] < epochs:
                    per_epoch = elapsed / time_data['epochs_done']
                    remaining = (epochs - time_data['epochs_done']) * per_epoch
                    time_data['remaining'] = remaining
                else:
                    time_data['remaining'] = 0

        epoch_progress = ((epoch + 1) / epochs) * 100
        if epoch_progress > 100:
            epoch_progress = 100
        if log_queue:
            log_queue.put(f"Epoch Progress: {epoch_progress:.2f}%\n")

    if log_queue and (not stop_flag or not stop_flag.is_set()):
        log_queue.put("Training completed.\n")

def softmax(x: np.ndarray) -> np.ndarray:

    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def run_inference(model: QuantumLanguageModel, input_sequence: List[int], token_to_id: Dict[str, int], id_to_token: Dict[int, str], max_length: int = 50, temperature: float = 1.0, log_callback=None):

    generated = input_sequence.copy()
    for _ in range(max_length):
        logits = model.forward(generated[-1])
        probabilities = softmax(logits / temperature)

        chosen_index = np.random.choice(len(probabilities), p=probabilities)
        generated.append(chosen_index)

        if chosen_index == token_to_id.get("<END>", chosen_index):
            break

    generated_tokens = [id_to_token.get(idx, "<UNK>") for idx in generated]
    response = " ".join(generated_tokens)

    if log_callback:
        log_callback(f"Generated Response:\n{response}\n\n")

    return generated_tokens, response

def quantum_loss(params: np.ndarray, model: QuantumLanguageModel, X: np.ndarray, Y: np.ndarray) -> float:

    model.set_all_parameters(params)
    
    total_loss = 0.0
    for input_id, target in zip(X, Y):
        logits = model.forward(input_id)
        loss = cross_entropy_loss(logits, target)
        total_loss += loss
    
    avg_loss = total_loss / len(X)
    return avg_loss

def train_quantum_model(model: QuantumLanguageModel, X: np.ndarray, Y: np.ndarray, epochs: int = 10, learning_rate: float = 0.001, num_threads: int = 1, log_queue: queue.Queue = None, stop_flag=None, time_lock: threading.Lock = None, time_data=None, optimizer=None):

    if time_data is None:
        time_data = {}
    start_time = time_data['start_time'] = time.time()
    time_data['epochs_done'] = 0
    time_data['epochs'] = epochs

    for epoch in range(epochs):
        if stop_flag and stop_flag.is_set():
            if log_queue:
                log_queue.put("Training stopped by user.\n")
            break

        if log_queue:
            log_queue.put(f"Starting Epoch {epoch+1}/{epochs}\n")

        epoch_start_time = time.time()

        def progress_callback(completed, total, param_index, gradient):

            progress = (completed / total) * 100
            if log_queue:
                log_queue.put(f"Gradient Computation Progress: {completed}/{total} parameters ({progress:.2f}%) completed.\n")

        gradients = compute_gradients_parallel(model, X, Y, num_processes=num_threads, progress_callback=progress_callback)

        if optimizer:
            updated_params = optimizer.step(gradients)
            model.set_all_parameters(updated_params)
        else:
            params = model._get_all_parameters()
            params -= learning_rate * gradients
            model.set_all_parameters(params)

        total_loss = np.mean([
            cross_entropy_loss(model.forward(x), y)
            for x, y in zip(X, Y)
        ])
        total_perplexity = np.mean([
            perplexity(model.forward(x), np.argmax(y))
            for x, y in zip(X, Y)
        ])

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        if log_queue:
            log_queue.put(f"Epoch {epoch+1}/{epochs} completed in {epoch_duration:.2f}s, Average Loss: {total_loss:.6f}, Perplexity: {total_perplexity:.6f}\n")

        if time_lock:
            with time_lock:
                time_data['epochs_done'] = epoch + 1
                elapsed = epoch_end_time - start_time
                if time_data['epochs_done'] > 0 and time_data['epochs_done'] < epochs:
                    per_epoch = elapsed / time_data['epochs_done']
                    remaining = (epochs - time_data['epochs_done']) * per_epoch
                    time_data['remaining'] = remaining
                else:
                    time_data['remaining'] = 0

        epoch_progress = ((epoch + 1) / epochs) * 100
        if epoch_progress > 100:
            epoch_progress = 100
        if log_queue:
            log_queue.put(f"Epoch Progress: {epoch_progress:.2f}%\n")

    if log_queue and (not stop_flag or not stop_flag.is_set()):
        log_queue.put("Training completed.\n")

def softmax(x: np.ndarray) -> np.ndarray:

    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def run_inference(model: QuantumLanguageModel, input_sequence: List[int], token_to_id: Dict[str, int], id_to_token: Dict[int, str], max_length: int = 50, temperature: float = 1.0, log_callback=None):

    generated = input_sequence.copy()
    for _ in range(max_length):
        logits = model.forward(generated[-1])
        probabilities = softmax(logits / temperature)


        chosen_index = np.random.choice(len(probabilities), p=probabilities)
        generated.append(chosen_index)

        if chosen_index == token_to_id.get("<END>", chosen_index):
            break

    generated_tokens = [id_to_token.get(idx, "<UNK>") for idx in generated]
    response = " ".join(generated_tokens)

    if log_callback:
        log_callback(f"Generated Response:\n{response}\n\n")

    return generated_tokens, response

def quantum_loss(params: np.ndarray, model: QuantumLanguageModel, X: np.ndarray, Y: np.ndarray) -> float:

    model.set_all_parameters(params)
    
    total_loss = 0.0
    for input_id, target in zip(X, Y):
        logits = model.forward(input_id)
        loss = cross_entropy_loss(logits, target)
        total_loss += loss
    
    avg_loss = total_loss / len(X)
    return avg_loss

def train_quantum_model(model: QuantumLanguageModel, X: np.ndarray, Y: np.ndarray, epochs: int = 10, learning_rate: float = 0.001, num_threads: int = 1, log_queue: queue.Queue = None, stop_flag=None, time_lock: threading.Lock = None, time_data=None, optimizer=None):

    if time_data is None:
        time_data = {}
    start_time = time_data['start_time'] = time.time()
    time_data['epochs_done'] = 0
    time_data['epochs'] = epochs

    for epoch in range(epochs):
        if stop_flag and stop_flag.is_set():
            if log_queue:
                log_queue.put("Training stopped by user.\n")
            break

        if log_queue:
            log_queue.put(f"Starting Epoch {epoch+1}/{epochs}\n")

        epoch_start_time = time.time()

        def progress_callback(completed, total, param_index, gradient):
            progress = (completed / total) * 100
            if log_queue:
                log_queue.put(f"Gradient Computation Progress: {completed}/{total} parameters ({progress:.2f}%) completed.\n")

        gradients = compute_gradients_parallel(model, X, Y, num_processes=num_threads, progress_callback=progress_callback)

        if optimizer:
            updated_params = optimizer.step(gradients)
            model.set_all_parameters(updated_params)
        else:
            params = model._get_all_parameters()
            params -= learning_rate * gradients
            model.set_all_parameters(params)

        total_loss = np.mean([
            cross_entropy_loss(model.forward(x), y)
            for x, y in zip(X, Y)
        ])
        total_perplexity = np.mean([
            perplexity(model.forward(x), np.argmax(y))
            for x, y in zip(X, Y)
        ])

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        if log_queue:
            log_queue.put(f"Epoch {epoch+1}/{epochs} completed in {epoch_duration:.2f}s, Average Loss: {total_loss:.6f}, Perplexity: {total_perplexity:.6f}\n")


        if time_lock:
            with time_lock:
                time_data['epochs_done'] = epoch + 1
                elapsed = epoch_end_time - start_time
                if time_data['epochs_done'] > 0 and time_data['epochs_done'] < epochs:
                    per_epoch = elapsed / time_data['epochs_done']
                    remaining = (epochs - time_data['epochs_done']) * per_epoch
                    time_data['remaining'] = remaining
                else:
                    time_data['remaining'] = 0

        epoch_progress = ((epoch + 1) / epochs) * 100
        if epoch_progress > 100:
            epoch_progress = 100
        if log_queue:
            log_queue.put(f"Epoch Progress: {epoch_progress:.2f}%\n")

    if log_queue and (not stop_flag or not stop_flag.is_set()):
        log_queue.put("Training completed.\n")

def softmax(x: np.ndarray) -> np.ndarray:

    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def run_inference(model: QuantumLanguageModel, input_sequence: List[int], token_to_id: Dict[str, int], id_to_token: Dict[int, str], max_length: int = 50, temperature: float = 1.0, log_callback=None):

    generated = input_sequence.copy()
    for _ in range(max_length):
        logits = model.forward(generated[-1])
        probabilities = softmax(logits / temperature)

        # Sample from the probability distribution
        chosen_index = np.random.choice(len(probabilities), p=probabilities)
        generated.append(chosen_index)
        if chosen_index == token_to_id.get("<END>", chosen_index):
            break

    generated_tokens = [id_to_token.get(idx, "<UNK>") for idx in generated]
    response = " ".join(generated_tokens)

    if log_callback:
        log_callback(f"Generated Response:\n{response}\n\n")

    return generated_tokens, response

def quantum_loss(params: np.ndarray, model: QuantumLanguageModel, X: np.ndarray, Y: np.ndarray) -> float:

    model.set_all_parameters(params)
    
    total_loss = 0.0
    for input_id, target in zip(X, Y):
        logits = model.forward(input_id)
        loss = cross_entropy_loss(logits, target)
        total_loss += loss
    
    avg_loss = total_loss / len(X)
    return avg_loss

def train_quantum_model(model: QuantumLanguageModel, X: np.ndarray, Y: np.ndarray, epochs: int = 10, learning_rate: float = 0.001, num_threads: int = 1, log_queue: queue.Queue = None, stop_flag=None, time_lock: threading.Lock = None, time_data=None, optimizer=None):

    if time_data is None:
        time_data = {}
    start_time = time_data['start_time'] = time.time()
    time_data['epochs_done'] = 0
    time_data['epochs'] = epochs

    for epoch in range(epochs):
        if stop_flag and stop_flag.is_set():
            if log_queue:
                log_queue.put("Training stopped by user.\n")
            break

        if log_queue:
            log_queue.put(f"Starting Epoch {epoch+1}/{epochs}\n")

        epoch_start_time = time.time()

        def progress_callback(completed, total, param_index, gradient):

            progress = (completed / total) * 100
            if log_queue:
                log_queue.put(f"Gradient Computation Progress: {completed}/{total} parameters ({progress:.2f}%) completed.\n")

        gradients = compute_gradients_parallel(model, X, Y, num_processes=num_threads, progress_callback=progress_callback)

        if optimizer:
            updated_params = optimizer.step(gradients)
            model.set_all_parameters(updated_params)
        else:

            params = model._get_all_parameters()
            params -= learning_rate * gradients
            model.set_all_parameters(params)

        total_loss = np.mean([
            cross_entropy_loss(model.forward(x), y)
            for x, y in zip(X, Y)
        ])
        total_perplexity = np.mean([
            perplexity(model.forward(x), np.argmax(y))
            for x, y in zip(X, Y)
        ])

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        if log_queue:
            log_queue.put(f"Epoch {epoch+1}/{epochs} completed in {epoch_duration:.2f}s, Average Loss: {total_loss:.6f}, Perplexity: {total_perplexity:.6f}\n")

        if time_lock:
            with time_lock:
                time_data['epochs_done'] = epoch + 1
                elapsed = epoch_end_time - start_time
                if time_data['epochs_done'] > 0 and time_data['epochs_done'] < epochs:
                    per_epoch = elapsed / time_data['epochs_done']
                    remaining = (epochs - time_data['epochs_done']) * per_epoch
                    time_data['remaining'] = remaining
                else:
                    time_data['remaining'] = 0

        epoch_progress = ((epoch + 1) / epochs) * 100
        if epoch_progress > 100:
            epoch_progress = 100
        if log_queue:
            log_queue.put(f"Epoch Progress: {epoch_progress:.2f}%\n")

    if log_queue and (not stop_flag or not stop_flag.is_set()):
        log_queue.put("Training completed.\n")


def softmax(x: np.ndarray) -> np.ndarray:

    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def run_inference(model: QuantumLanguageModel, input_sequence: List[int], token_to_id: Dict[str, int], id_to_token: Dict[int, str], max_length: int = 50, temperature: float = 1.0, log_callback=None):

    generated = input_sequence.copy()
    for _ in range(max_length):
        logits = model.forward(generated[-1])
        probabilities = softmax(logits / temperature)

        chosen_index = np.random.choice(len(probabilities), p=probabilities)
        generated.append(chosen_index)

        if chosen_index == token_to_id.get("<END>", chosen_index):
            break

    generated_tokens = [id_to_token.get(idx, "<UNK>") for idx in generated]
    response = " ".join(generated_tokens)

    if log_callback:
        log_callback(f"Generated Response:\n{response}\n\n")

    return generated_tokens, response

class QELM_GUI:

    def __init__(self, master):
        try:
            self.master = master
            master.title("QELM Trainer - Advanced Integration")
            master.geometry("1800x1100")  # Increased window size for better layout
            master.resizable(False, False)

            self.vocab_size = 10000  # Default vocabulary size
            self.embed_dim = 256      # Default embedding dimension
            self.num_layers = 2       # Default number of stacked layers
            self.neurons_per_layer = 4  # Default number of neurons per layer
            self.sim_method = 'cpu'
            self.num_threads = min(8, multiprocessing.cpu_count())  # Adjusted for higher model complexity
            self.model = QuantumLanguageModel(
                vocab_size=self.vocab_size,
                embed_dim=self.embed_dim,
                num_layers=self.num_layers,
                neurons_per_layer=self.neurons_per_layer,
                sim_method=self.sim_method,
                num_threads=self.num_threads,
                enable_logging=True
            )

            self.token_to_id = {}
            self.id_to_token = {}

            self.optimizer = AdamOptimizer(self.model._get_all_parameters(), lr=0.001)

            self.stop_flag = threading.Event()
            self.time_data = {'start_time': 0, 'epochs_done': 0, 'remaining': 0, 'epochs': 0}
            self.time_lock = threading.Lock()

            if psutil:
                self.process = psutil.Process(os.getpid())
                self.process.cpu_percent(interval=None)  # Initialize
            else:
                self.process = None

            log_dir = r'C:\Users\'  
            os.makedirs(log_dir, exist_ok=True)
            logging.basicConfig(
                filename=os.path.join(log_dir, 'qelm_integrated.log'),
                filemode='a',
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s'
            )

            console = logging.StreamHandler()
            console.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            console.setFormatter(formatter)
            logging.getLogger('').addHandler(console)

            self.master.configure(bg="#2C3E50")
            style = ttk.Style(self.master)
            style.theme_use('clam')
            style.configure(".", background="#2C3E50", foreground="white")
            style.configure("TFrame", background="#2C3E50")
            style.configure("TLabelFrame", background="#34495E", foreground="white")
            style.configure("TLabel", background="#2C3E50", foreground="white")
            style.configure("TButton", background="#34495E", foreground="white", padding=6, relief="flat")
            style.configure("TNotebook", background="#2C3E50")
            style.configure("TNotebook.Tab", background="#34495E", foreground="white")
            style.configure("Horizontal.TProgressbar", background="#1ABC9C", troughcolor="#34495E")

            style.configure("Custom.TEntry", fieldbackground="#455A64", foreground="white", insertcolor="white")
            style.configure("TSpinbox", fieldbackground="#455A64", foreground="white")
            style.map("TButton", foreground=[('active', 'white')], background=[('active', '#1F2A36')])

            self.create_widgets()
            self.update_resource_usage()
            self.update_time_label()

            self.log_queue = queue.Queue()
            self.master.after(100, self.process_log_queue)

        except Exception as e:
            error_trace = traceback.format_exc()
            logging.critical(f"GUI Initialization error:\n{error_trace}")
            messagebox.showerror("Initialization Error", f"An error occurred during GUI initialization:\n{e}\n\nCheck the log file for more details.")
            sys.exit(1)

    def create_widgets(self):

        container = ttk.Frame(self.master)
        container.pack(fill='both', expand=True)

        left_frame = ttk.Frame(container)
        left_frame.pack(side='left', fill='both', expand=True, padx=10, pady=10)

        right_frame = ttk.Frame(container)
        right_frame.pack(side='right', fill='y', padx=10, pady=10)

        self.notebook = ttk.Notebook(left_frame)
        self.tab_train = ttk.Frame(self.notebook)
        self.tab_infer = ttk.Frame(self.notebook)
        self.tab_manage = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_train, text='Train Model')
        self.notebook.add(self.tab_infer, text='Run Inference')
        self.notebook.add(self.tab_manage, text='Manage Token Mappings')
        self.notebook.pack(fill='both', expand=True)

        dataset_frame = ttk.LabelFrame(self.tab_train, text="Dataset Selection")
        dataset_frame.pack(fill='x', padx=10, pady=10)

        self.dataset_path_var = tk.StringVar(value="No dataset selected.")
        ttk.Label(dataset_frame, textvariable=self.dataset_path_var).pack(side='left', padx=10, pady=10)
        select_dataset_btn = ttk.Button(dataset_frame, text="Select Dataset", command=self.select_dataset)
        select_dataset_btn.pack(side='right', padx=10, pady=10)

        hyperparams_frame = ttk.LabelFrame(self.tab_train, text="Model Parameters")
        hyperparams_frame.pack(fill='x', padx=10, pady=10)

        ttk.Label(hyperparams_frame, text="Vocabulary Size:").grid(row=0, column=0, padx=10, pady=10, sticky='e')
        self.vocab_size_entry = ttk.Entry(hyperparams_frame, width=15, style="Custom.TEntry")
        self.vocab_size_entry.insert(0, str(self.vocab_size))
        self.vocab_size_entry.grid(row=0, column=1, padx=10, pady=10, sticky='w')

        ttk.Label(hyperparams_frame, text="Embedding Dimension:").grid(row=1, column=0, padx=10, pady=10, sticky='e')
        self.embed_dim_entry = ttk.Entry(hyperparams_frame, width=15, style="Custom.TEntry")
        self.embed_dim_entry.insert(0, str(self.embed_dim))
        self.embed_dim_entry.grid(row=1, column=1, padx=10, pady=10, sticky='w')

        ttk.Label(hyperparams_frame, text="Number of Stacked Layers:").grid(row=2, column=0, padx=10, pady=10, sticky='e')
        self.num_layers_entry = ttk.Entry(hyperparams_frame, width=15, style="Custom.TEntry")
        self.num_layers_entry.insert(0, str(self.num_layers))
        self.num_layers_entry.grid(row=2, column=1, padx=10, pady=10, sticky='w')

        ttk.Label(hyperparams_frame, text="Neurons per Layer:").grid(row=3, column=0, padx=10, pady=10, sticky='e')
        self.neurons_per_layer_entry = ttk.Entry(hyperparams_frame, width=15, style="Custom.TEntry")
        self.neurons_per_layer_entry.insert(0, str(self.neurons_per_layer))
        self.neurons_per_layer_entry.grid(row=3, column=1, padx=10, pady=10, sticky='w')

        ttk.Label(hyperparams_frame, text="Learning Rate:").grid(row=4, column=0, padx=10, pady=10, sticky='e')
        self.lr_entry = ttk.Entry(hyperparams_frame, width=15, style="Custom.TEntry")
        self.lr_entry.insert(0, "0.001")
        self.lr_entry.grid(row=4, column=1, padx=10, pady=10, sticky='w')

        ttk.Label(hyperparams_frame, text="Epochs:").grid(row=5, column=0, padx=10, pady=10, sticky='e')
        self.epochs_entry = ttk.Entry(hyperparams_frame, width=15, style="Custom.TEntry")
        self.epochs_entry.insert(0, "2")  # Reduced epochs for testing
        self.epochs_entry.grid(row=5, column=1, padx=10, pady=10, sticky='w')

        sim_settings_frame = ttk.LabelFrame(self.tab_train, text="Simulation Settings")
        sim_settings_frame.pack(fill='x', padx=10, pady=10)

        ttk.Label(sim_settings_frame, text="Simulation Method:").grid(row=0, column=0, padx=10, pady=10, sticky='e')
        self.sim_method_var = tk.StringVar(value="cpu")
        cpu_radio = ttk.Radiobutton(sim_settings_frame, text='CPU', variable=self.sim_method_var, value='cpu', command=self.update_threads_based_on_method)
        gpu_radio = ttk.Radiobutton(sim_settings_frame, text='GPU', variable=self.sim_method_var, value='gpu', command=self.update_threads_based_on_method)
        cpu_radio.grid(row=0, column=1, padx=10, pady=10, sticky='w')
        gpu_radio.grid(row=0, column=2, padx=10, pady=10, sticky='w')

        ttk.Label(sim_settings_frame, text="Number of Threads:").grid(row=1, column=0, padx=10, pady=10, sticky='e')
        self.num_threads_var = tk.IntVar(value=self.num_threads)
        self.num_threads_spinbox = ttk.Spinbox(
            sim_settings_frame,
            from_=1,
            to=multiprocessing.cpu_count(),
            textvariable=self.num_threads_var,
            width=5
        )
        self.num_threads_spinbox.grid(row=1, column=1, padx=10, pady=10, sticky='w')
        ttk.Label(sim_settings_frame, text=f"(Max: {multiprocessing.cpu_count()})").grid(row=1, column=2, padx=10, pady=10, sticky='w')

        train_controls_frame = ttk.Frame(self.tab_train)
        train_controls_frame.pack(fill='x', padx=10, pady=10)

        self.train_button = ttk.Button(train_controls_frame, text="Start Training", command=self.train_model)
        self.train_button.pack(side='left', padx=10, pady=10)

        stop_button = ttk.Button(train_controls_frame, text="STOP (Graceful)", command=self.stop_training)
        stop_button.pack(side='left', padx=10, pady=10)

        hard_stop_button = ttk.Button(train_controls_frame, text="HARD STOP (Immediate)", command=self.hard_stop)
        hard_stop_button.pack(side='left', padx=10, pady=10)

        self.save_button = ttk.Button(train_controls_frame, text="Save Model", command=self.save_model)
        self.save_button.pack(side='left', padx=10, pady=10)

        self.load_button = ttk.Button(train_controls_frame, text="Load Model", command=self.load_model)
        self.load_button.pack(side='left', padx=10, pady=10)

        progress_bars_frame = ttk.Frame(self.tab_train)
        progress_bars_frame.pack(fill='x', padx=10, pady=10)

        ttk.Label(progress_bars_frame, text="Training Progress:").pack(anchor='w', padx=10, pady=5)
        self.epoch_progress = ttk.Progressbar(progress_bars_frame, orient='horizontal', length=800, mode='determinate')
        self.epoch_progress.pack(fill='x', padx=10, pady=5)

        ttk.Label(progress_bars_frame, text="Gradient Computation Progress:").pack(anchor='w', padx=10, pady=5)
        self.gradient_progress = ttk.Progressbar(progress_bars_frame, orient='horizontal', length=800, mode='determinate')
        self.gradient_progress.pack(fill='x', padx=10, pady=5)

        log_frame = ttk.LabelFrame(self.tab_train, text="Training Log")
        log_frame.pack(fill='both', expand=True, padx=10, pady=10)

        self.train_log = scrolledtext.ScrolledText(log_frame, state='disabled', wrap='word', font=("Courier", 10),
                                                 bg="#2C3E50", fg="white", insertbackground="white")
        self.train_log.pack(fill='both', expand=True, padx=5, pady=5)

        eval_metrics_frame = ttk.LabelFrame(self.tab_train, text="Evaluation Metrics")
        eval_metrics_frame.pack(fill='x', padx=10, pady=10)

        self.perplexity_label = ttk.Label(eval_metrics_frame, text="Perplexity: N/A")
        self.perplexity_label.pack(anchor='w', padx=10, pady=5)

        self.bleu_label = ttk.Label(eval_metrics_frame, text="BLEU Score: N/A")
        self.bleu_label.pack(anchor='w', padx=10, pady=5)

        self.fidelity_label = ttk.Label(eval_metrics_frame, text="Fidelity: N/A")
        self.fidelity_label.pack(anchor='w', padx=10, pady=5)

        inference_frame = ttk.LabelFrame(self.tab_infer, text="Inference")
        inference_frame.pack(fill='x', padx=10, pady=10)

        ttk.Label(inference_frame, text="Input Token:").grid(row=0, column=0, padx=10, pady=10, sticky='e')
        self.input_token_entry = ttk.Entry(inference_frame, width=30, style="Custom.TEntry")
        self.input_token_entry.grid(row=0, column=1, padx=10, pady=10, sticky='w')

        ttk.Label(inference_frame, text="Max Length:").grid(row=1, column=0, padx=10, pady=10, sticky='e')
        self.max_length_entry = ttk.Entry(inference_frame, width=15, style="Custom.TEntry")
        self.max_length_entry.insert(0, "50")
        self.max_length_entry.grid(row=1, column=1, padx=10, pady=10, sticky='w')

        ttk.Label(inference_frame, text="Temperature:").grid(row=2, column=0, padx=10, pady=10, sticky='e')
        self.temperature_entry = ttk.Entry(inference_frame, width=15, style="Custom.TEntry")
        self.temperature_entry.insert(0, "1.0")
        self.temperature_entry.grid(row=2, column=1, padx=10, pady=10, sticky='w')

        inference_controls_frame = ttk.Frame(self.tab_infer)
        inference_controls_frame.pack(fill='x', padx=10, pady=10)

        self.infer_button = ttk.Button(inference_controls_frame, text="Run Inference", command=self.run_inference)
        self.infer_button.pack(side='left', padx=10, pady=10)

        infer_log_frame = ttk.LabelFrame(self.tab_infer, text="Inference Output")
        infer_log_frame.pack(fill='both', expand=True, padx=10, pady=10)

        self.infer_log = scrolledtext.ScrolledText(infer_log_frame, state='disabled', wrap='word', font=("Courier", 10),
                                                 bg="#2C3E50", fg="white", insertbackground="white")
        self.infer_log.pack(fill='both', expand=True, padx=5, pady=5)

        token_map_frame = ttk.LabelFrame(self.tab_manage, text="Token Mappings")
        token_map_frame.pack(fill='both', expand=True, padx=10, pady=10)

        load_token_map_button = ttk.Button(token_map_frame, text="Load Token Map", command=self.load_token_map)
        load_token_map_button.pack(side='top', padx=10, pady=10)

        self.token_map_display = scrolledtext.ScrolledText(token_map_frame, state='disabled', wrap='word', font=("Courier", 10),
                                                          bg="#2C3E50", fg="white", insertbackground="white")
        self.token_map_display.pack(fill='both', expand=True, padx=5, pady=5)

        usage_frame = ttk.LabelFrame(right_frame, text="System Resources & Time")
        usage_frame.pack(fill='y', padx=5, pady=5)

        self.cpu_label = ttk.Label(usage_frame, text="CPU: N/A")
        self.cpu_label.pack(anchor='w', padx=10, pady=5)

        self.gpu_label = ttk.Label(usage_frame, text="GPU: Check externally (e.g., nvidia-smi)")
        self.gpu_label.pack(anchor='w', padx=10, pady=5)

        self.time_label = ttk.Label(usage_frame, text="Elapsed: 0s | Remaining: Estimating...")
        self.time_label.pack(anchor='w', padx=10, pady=5)

    def process_log_queue(self):

        try:
            while not self.log_queue.empty():
                message = self.log_queue.get_nowait()
                self.train_log.config(state='normal')
                self.train_log.insert(tk.END, message)
                self.train_log.see(tk.END)
                self.train_log.config(state='disabled')
        except Exception as e:
            logging.error(f"Error processing log queue: {e}")
        finally:
            self.master.after(100, self.process_log_queue)  # Continue polling

    def update_threads_based_on_method(self):

        method = self.sim_method_var.get()
        max_threads = multiprocessing.cpu_count()
        self.num_threads_spinbox.config(to=max_threads)
        if self.num_threads_var.get() > max_threads:
            self.num_threads_var.set(max_threads)

    def log_train(self, message: str):

        if hasattr(self, 'log_queue'):
            self.log_queue.put(message)

    def log_infer(self, message: str):

        self.infer_log.config(state='normal')
        self.infer_log.insert(tk.END, message)
        self.infer_log.see(tk.END)
        self.infer_log.config(state='disabled')

    def log_token_map(self, message: str):

        self.token_map_display.config(state='normal')
        self.token_map_display.insert(tk.END, message)
        self.token_map_display.see(tk.END)
        self.token_map_display.config(state='disabled')

    def select_dataset(self):

        try:
            file_path = filedialog.askopenfilename(title="Select Dataset File",
                                                   filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
            if file_path:
                self.dataset_path = file_path
                self.dataset_path_var.set(file_path)
                self.log_train(f"Selected Dataset: {file_path}\n")
                self.token_to_id = {}
                self.id_to_token = {}
        except Exception as e:
            err_msg = f"Error selecting dataset:\n{traceback.format_exc()}"
            self.log_train(err_msg + "\n")
            messagebox.showerror("Dataset Selection Error", err_msg)

    def train_model(self):

        try:
            vocab_size = int(self.vocab_size_entry.get())
            embed_dim = int(self.embed_dim_entry.get())
            num_layers = int(self.num_layers_entry.get())
            neurons_per_layer = int(self.neurons_per_layer_entry.get())
            lr = float(self.lr_entry.get())
            epochs = int(self.epochs_entry.get())

            if vocab_size <= 0 or embed_dim <= 0 or num_layers <= 0 or neurons_per_layer <= 0 or lr <= 0 or epochs <= 0:
                raise ValueError

            if embed_dim % (neurons_per_layer * 20) != 0:
                messagebox.showerror("Invalid Input", "Embedding Dimension must be divisible by (Neurons per Layer * 20).")
                return

        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid positive integers for model parameters and epochs, and a positive float for learning rate.")
            return

        sim_method = self.sim_method_var.get()
        num_threads = self.num_threads_var.get()
        max_threads = multiprocessing.cpu_count()
        if num_threads > max_threads:
            messagebox.showwarning("Thread Limit", f"Resetting threads to max {max_threads}")
            num_threads = max_threads
            self.num_threads_var.set(num_threads)

        if hasattr(self, 'dataset_path') and self.dataset_path:
            dataset_path = self.dataset_path
            try:
                X, Y, token_to_id = load_real_dataset(dataset_path, vocab_size)
                self.X = X
                self.Y = Y
                self.token_to_id = token_to_id
                self.id_to_token = {idx: token for token, idx in token_to_id.items()}
                self.log_train(f"Loaded real dataset from {dataset_path}\n")
            except Exception as e:
                err_msg = f"Failed to load dataset:\n{traceback.format_exc()}"
                self.log_train(err_msg + "\n")
                messagebox.showerror("Dataset Load Error", err_msg)
                return
        else:
            X, Y = create_synthetic_dataset(vocab_size, num_samples=500)  # Increased samples for better training
            self.X = X
            self.Y = Y
            self.log_train("Using synthetic dataset for training.\n")

        try:
            self.model = QuantumLanguageModel(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                num_layers=num_layers,
                neurons_per_layer=neurons_per_layer,
                sim_method=sim_method,
                num_threads=num_threads,
                enable_logging=True
            )
            self.optimizer = AdamOptimizer(self.model._get_all_parameters(), lr=lr)
            self.log_train("Model re-initialized with new parameters.\n")
        except Exception as e:
            err_msg = f"Failed to initialize model with new parameters:\n{traceback.format_exc()}"
            self.log_train(err_msg + "\n")
            messagebox.showerror("Model Initialization Error", err_msg)
            return

        self.model.sim_method = sim_method
        self.model.num_threads = num_threads
        self.model.backend = self.model.initialize_simulator()

        self.train_button.config(state='disabled')
        self.save_button.config(state='disabled')
        self.load_button.config(state='disabled')
        self.infer_button.config(state='disabled')
        self.stop_flag.clear()

        self.epoch_progress['value'] = 0
        self.gradient_progress['value'] = 0
        self.train_log.config(state='normal')
        self.train_log.delete('1.0', tk.END)
        self.train_log.config(state='disabled')
        self.log_train("Starting training...\n")

        with self.time_lock:
            self.time_data['start_time'] = time.time()
            self.time_data['epochs_done'] = 0
            self.time_data['epochs'] = epochs
            self.time_data['remaining'] = 0

        training_thread = threading.Thread(target=self.training_process, args=(epochs, num_threads), daemon=True)
        training_thread.start()

    def training_process(self, epochs: int, num_threads: int):

        try:
            self.log_train("Training thread started.\n")
            train_quantum_model(
                self.model,
                self.X,
                self.Y,
                epochs=epochs,
                learning_rate=self.optimizer.lr,
                num_threads=num_threads,
                log_queue=self.log_queue,
                stop_flag=self.stop_flag,
                time_lock=self.time_lock,
                time_data=self.time_data,
                optimizer=self.optimizer
            )
            if not self.stop_flag.is_set():
                self.log_train("Training completed successfully.\n")
                messagebox.showinfo("Training Completed", "Model training completed successfully.")
        except Exception as e:
            err_msg = f"Training error:\n{traceback.format_exc()}"
            self.log_train(err_msg + "\n")
            messagebox.showerror("Training Error", err_msg)
        finally:
            self.train_button.config(state='normal')
            self.save_button.config(state='normal')
            self.load_button.config(state='normal')
            self.infer_button.config(state='normal')
            if not self.stop_flag.is_set():
                self.epoch_progress['value'] = 100
                self.gradient_progress['value'] = 100

            self.evaluate_model()

    def stop_training(self):

        self.stop_flag.set()
        self.log_train("Stop signal sent. Will stop after current epoch.\n")

    def hard_stop(self):

        self.log_train("Hard stop invoked. Terminating immediately.\n")
        os._exit(1)

    def save_model(self):

        try:
            save_path = filedialog.asksaveasfilename(title="Save Model", defaultextension=".qelm",
                                                     filetypes=[("QELM Files", "*.qelm"), ("All Files", "*.*")])
            if save_path:
                self.model.save_model(save_path)
                if self.token_to_id:
                    token_map_path = save_path.replace(".qelm", "_token_map.json")
                    with open(token_map_path, 'w') as f:
                        json.dump(self.token_to_id, f, indent=4)
                    self.log_train(f"Token mappings saved to {token_map_path}\n")
                messagebox.showinfo("Model Saved", f"Model saved to {save_path}")
        except Exception as e:
            err_msg = f"Save model error:\n{traceback.format_exc()}"
            self.log_train(err_msg + "\n")
            messagebox.showerror("Save Error", err_msg)

    def load_model(self):

        try:
            load_path = filedialog.askopenfilename(title="Load Model",
                                                  filetypes=[("QELM Files", "*.qelm"), ("All Files", "*.*")])
            if load_path:
                self.model.load_model(load_path)
                token_map_path = load_path.replace(".qelm", "_token_map.json")
                try:
                    with open(token_map_path, 'r') as f:
                        self.token_to_id = json.load(f)
                    self.id_to_token = {int(idx): token for token, idx in self.token_to_id.items()}
                    self.log_train(f"Loaded token mappings from {token_map_path}\n")
                    self.display_token_map()
                except FileNotFoundError:
                    self.log_train("No token mappings file found.\n")
                messagebox.showinfo("Model Loaded", f"Model loaded from {load_path}")
        except Exception as e:
            err_msg = f"Load model error:\n{traceback.format_exc()}"
            self.log_train(err_msg + "\n")
            messagebox.showerror("Load Error", err_msg)

    def run_inference(self):

        input_token = self.input_token_entry.get().strip().lower()
        if not input_token:
            messagebox.showerror("Input Error", "Please enter an input token for inference.")
            return

        try:
            max_length = int(self.max_length_entry.get())
            temperature = float(self.temperature_entry.get())
            if max_length <= 0 or temperature <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter positive values for max length and temperature.")
            return

        self.infer_button.config(state='disabled')
        self.log_infer(f"Running inference for '{input_token}' with max_length={max_length} and temperature={temperature}...\n")

        inference_thread = threading.Thread(target=self.inference_process, args=(input_token, max_length, temperature), daemon=True)
        inference_thread.start()

    def inference_process(self, input_token: str, max_length: int, temperature: float):

        try:
            if input_token not in self.token_to_id:
                raise ValueError(f"Input token '{input_token}' not found in token mappings.")
            input_id = self.token_to_id[input_token]
            generated_tokens, response = run_inference(
                self.model,
                [input_id],
                self.token_to_id,
                self.id_to_token,
                max_length=max_length,
                temperature=temperature,
                log_callback=self.log_infer
            )
            messagebox.showinfo("Inference Completed", "Inference completed successfully.")
        except Exception as e:
            err_msg = f"Inference error:\n{traceback.format_exc()}"
            self.log_infer(err_msg + "\n")
            messagebox.showerror("Inference Error", err_msg)
        finally:
            self.infer_button.config(state='normal')

    def load_token_map(self):

        try:
            file_path = filedialog.askopenfilename(title="Load Token Map",
                                                   filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")])
            if file_path:
                with open(file_path, 'r') as f:
                    self.token_to_id = json.load(f)
                self.id_to_token = {int(idx): token for token, idx in self.token_to_id.items()}
                self.log_token_map(f"Loaded token mappings from {file_path}\n")
                self.display_token_map()
                messagebox.showinfo("Token Map Loaded", f"Token mappings loaded from {file_path}")
        except Exception as e:
            err_msg = f"Load token map error:\n{traceback.format_exc()}"
            self.log_token_map(err_msg + "\n")
            messagebox.showerror("Load Error", err_msg)

    def display_token_map(self):

        self.token_map_display.config(state='normal')
        self.token_map_display.delete('1.0', tk.END)
        self.token_map_display.insert(tk.END, "Token Mappings (Token: ID):\n\n")
        for token, idx in sorted(self.token_to_id.items(), key=lambda x: x[1]):
            self.token_map_display.insert(tk.END, f"{token}: {idx}\n")
        self.token_map_display.config(state='disabled')

    def update_resource_usage(self):

        if self.process:
            cpu_usage = f"{self.process.cpu_percent(interval=None)}%"
        else:
            cpu_usage = "psutil not installed"

        self.cpu_label.config(text=f"CPU: {cpu_usage}")
        self.gpu_label.config(text=f"GPU: Check externally (e.g., nvidia-smi)")

        self.master.after(1000, self.update_resource_usage)

    def update_time_label(self):

        with self.time_lock:
            elapsed = time.time() - self.time_data.get('start_time', 0)
            elapsed_str = f"{elapsed:.1f}s"

            remaining = self.time_data.get('remaining', 0)
            if remaining > 0:
                remaining_str = f"{remaining:.1f}s"
            else:
                remaining_str = "Estimating..."

        self.time_label.config(text=f"Elapsed: {elapsed_str} | Remaining: {remaining_str}")

        # Update every second
        self.master.after(1000, self.update_time_label)

    def evaluate_model(self):

        perplexities = []
        for x, y in zip(self.X, self.Y):
            logits = self.model.forward(x)
            perp = perplexity(logits, np.argmax(y))
            perplexities.append(perp)
        avg_perplexity = np.mean(perplexities)

        hypotheses = []
        references = []
        for x, y in zip(self.X, self.Y):
            logits = self.model.forward(x)
            predicted = np.argmax(logits)
            hypotheses.append([self.id_to_token.get(predicted, "<UNK>")])
            references.append([self.id_to_token.get(np.argmax(y), "<UNK>")])

        bleu_scores = []
        for ref, hyp in zip(references, hypotheses):
            bleu = bleu_score(ref, hyp)
            bleu_scores.append(bleu)
        avg_bleu = np.mean(bleu_scores)

        fidelity_scores = []
        for x, y in zip(self.X[:100], self.Y[:100]):  # Limit to first 100 for performance
            fidelity = self.model.simulate_and_evaluate(x, np.argmax(y))
            fidelity_scores.append(fidelity)
        avg_fidelity = np.mean(fidelity_scores)

        self.perplexity_label.config(text=f"Perplexity: {avg_perplexity:.4f}")
        self.bleu_label.config(text=f"BLEU Score: {avg_bleu:.4f}")
        self.fidelity_label.config(text=f"Fidelity: {avg_fidelity:.4f}")

    def run_inference(self):

        input_token = self.input_token_entry.get().strip().lower()
        if not input_token:
            messagebox.showerror("Input Error", "Please enter an input token for inference.")
            return

        try:
            max_length = int(self.max_length_entry.get())
            temperature = float(self.temperature_entry.get())
            if max_length <= 0 or temperature <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter positive values for max length and temperature.")
            return

        self.infer_button.config(state='disabled')
        self.log_infer(f"Running inference for '{input_token}' with max_length={max_length} and temperature={temperature}...\n")

        inference_thread = threading.Thread(target=self.inference_process, args=(input_token, max_length, temperature), daemon=True)
        inference_thread.start()

    def inference_process(self, input_token: str, max_length: int, temperature: float):

        try:
            if input_token not in self.token_to_id:
                raise ValueError(f"Input token '{input_token}' not found in token mappings.")
            input_id = self.token_to_id[input_token]
            generated_tokens, response = run_inference(
                self.model,
                [input_id],
                self.token_to_id,
                self.id_to_token,
                max_length=max_length,
                temperature=temperature,
                log_callback=self.log_infer
            )
            messagebox.showinfo("Inference Completed", "Inference completed successfully.")
        except Exception as e:
            err_msg = f"Inference error:\n{traceback.format_exc()}"
            self.log_infer(err_msg + "\n")
            messagebox.showerror("Inference Error", err_msg)
        finally:
            self.infer_button.config(state='normal')

    def load_token_map(self):

        try:
            file_path = filedialog.askopenfilename(title="Load Token Map",
                                                   filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")])
            if file_path:
                with open(file_path, 'r') as f:
                    self.token_to_id = json.load(f)
                self.id_to_token = {int(idx): token for token, idx in self.token_to_id.items()}
                self.log_token_map(f"Loaded token mappings from {file_path}\n")
                self.display_token_map()
                messagebox.showinfo("Token Map Loaded", f"Token mappings loaded from {file_path}")
        except Exception as e:
            err_msg = f"Load token map error:\n{traceback.format_exc()}"
            self.log_token_map(err_msg + "\n")
            messagebox.showerror("Load Error", err_msg)

    def display_token_map(self):

        self.token_map_display.config(state='normal')
        self.token_map_display.delete('1.0', tk.END)
        self.token_map_display.insert(tk.END, "Token Mappings (Token: ID):\n\n")
        for token, idx in sorted(self.token_to_id.items(), key=lambda x: x[1]):
            self.token_map_display.insert(tk.END, f"{token}: {idx}\n")
        self.token_map_display.config(state='disabled')

    def update_resource_usage(self):

        if self.process:
            cpu_usage = f"{self.process.cpu_percent(interval=None)}%"
        else:
            cpu_usage = "psutil not installed"

        self.cpu_label.config(text=f"CPU: {cpu_usage}")
        self.gpu_label.config(text=f"GPU: Check externally (e.g., nvidia-smi)")

        self.master.after(1000, self.update_resource_usage)

    def update_time_label(self):
      
        with self.time_lock:
            elapsed = time.time() - self.time_data.get('start_time', 0)
            elapsed_str = f"{elapsed:.1f}s"

            remaining = self.time_data.get('remaining', 0)
            if remaining > 0:
                remaining_str = f"{remaining:.1f}s"
            else:
                remaining_str = "Estimating..."

        self.time_label.config(text=f"Elapsed: {elapsed_str} | Remaining: {remaining_str}")

        # Update every second
        self.master.after(1000, self.update_time_label)

    def evaluate_model(self):

        perplexities = []
        for x, y in zip(self.X, self.Y):
            logits = self.model.forward(x)
            perp = perplexity(logits, np.argmax(y))
            perplexities.append(perp)
        avg_perplexity = np.mean(perplexities)

        hypotheses = []
        references = []
        for x, y in zip(self.X, self.Y):
            logits = self.model.forward(x)
            predicted = np.argmax(logits)
            hypotheses.append([self.id_to_token.get(predicted, "<UNK>")])
            references.append([self.id_to_token.get(np.argmax(y), "<UNK>")])

        bleu_scores = []
        for ref, hyp in zip(references, hypotheses):
            bleu = bleu_score(ref, hyp)
            bleu_scores.append(bleu)
        avg_bleu = np.mean(bleu_scores)

        fidelity_scores = []
        for x, y in zip(self.X[:100], self.Y[:100]):  # Limit to first 100 for performance
            fidelity = self.model.simulate_and_evaluate(x, np.argmax(y))
            fidelity_scores.append(fidelity)
        avg_fidelity = np.mean(fidelity_scores)

        self.perplexity_label.config(text=f"Perplexity: {avg_perplexity:.4f}")
        self.bleu_label.config(text=f"BLEU Score: {avg_bleu:.4f}")
        self.fidelity_label.config(text=f"Fidelity: {avg_fidelity:.4f}")

    def run_inference(self):

        input_token = self.input_token_entry.get().strip().lower()
        if not input_token:
            messagebox.showerror("Input Error", "Please enter an input token for inference.")
            return

        try:
            max_length = int(self.max_length_entry.get())
            temperature = float(self.temperature_entry.get())
            if max_length <= 0 or temperature <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter positive values for max length and temperature.")
            return

        self.infer_button.config(state='disabled')
        self.log_infer(f"Running inference for '{input_token}' with max_length={max_length} and temperature={temperature}...\n")

        # Start inference in a separate thread
        inference_thread = threading.Thread(target=self.inference_process, args=(input_token, max_length, temperature), daemon=True)
        inference_thread.start()

    def inference_process(self, input_token: str, max_length: int, temperature: float):

        try:
            if input_token not in self.token_to_id:
                raise ValueError(f"Input token '{input_token}' not found in token mappings.")
            input_id = self.token_to_id[input_token]
            generated_tokens, response = run_inference(
                self.model,
                [input_id],
                self.token_to_id,
                self.id_to_token,
                max_length=max_length,
                temperature=temperature,
                log_callback=self.log_infer
            )
            messagebox.showinfo("Inference Completed", "Inference completed successfully.")
        except Exception as e:
            err_msg = f"Inference error:\n{traceback.format_exc()}"
            self.log_infer(err_msg + "\n")
            messagebox.showerror("Inference Error", err_msg)
        finally:
            self.infer_button.config(state='normal')

    def load_token_map(self):

        try:
            file_path = filedialog.askopenfilename(title="Load Token Map",
                                                   filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")])
            if file_path:
                with open(file_path, 'r') as f:
                    self.token_to_id = json.load(f)
                self.id_to_token = {int(idx): token for token, idx in self.token_to_id.items()}
                self.log_token_map(f"Loaded token mappings from {file_path}\n")
                self.display_token_map()
                messagebox.showinfo("Token Map Loaded", f"Token mappings loaded from {file_path}")
        except Exception as e:
            err_msg = f"Load token map error:\n{traceback.format_exc()}"
            self.log_token_map(err_msg + "\n")
            messagebox.showerror("Load Error", err_msg)

    def display_token_map(self):

        self.token_map_display.config(state='normal')
        self.token_map_display.delete('1.0', tk.END)
        self.token_map_display.insert(tk.END, "Token Mappings (Token: ID):\n\n")
        for token, idx in sorted(self.token_to_id.items(), key=lambda x: x[1]):
            self.token_map_display.insert(tk.END, f"{token}: {idx}\n")
        self.token_map_display.config(state='disabled')

    def update_resource_usage(self):

        if self.process:
            cpu_usage = f"{self.process.cpu_percent(interval=None)}%"
        else:
            cpu_usage = "psutil not installed"

        self.cpu_label.config(text=f"CPU: {cpu_usage}")
        self.gpu_label.config(text=f"GPU: Check externally (e.g., nvidia-smi)")

        self.master.after(1000, self.update_resource_usage)

    def update_time_label(self):
      elapsed and remaining training time.
        with self.time_lock:
            elapsed = time.time() - self.time_data.get('start_time', 0)
            elapsed_str = f"{elapsed:.1f}s"

            remaining = self.time_data.get('remaining', 0)
            if remaining > 0:
                remaining_str = f"{remaining:.1f}s"
            else:
                remaining_str = "Estimating..."

        self.time_label.config(text=f"Elapsed: {elapsed_str} | Remaining: {remaining_str}")

        # Update every second
        self.master.after(1000, self.update_time_label)

def main():

    try:
        root = tk.Tk()
        gui = QELM_GUI(root)
        root.mainloop()
    except Exception as e:
        error_trace = traceback.format_exc()
        logging.critical(f"Unexpected error:\n{error_trace}")
        # Create a hidden root to display the message box
        hidden_root = tk.Tk()
        hidden_root.withdraw()
        messagebox.showerror("Unexpected Error", f"An unexpected error occurred:\n{e}\n\nCheck the log file for more details.")
        sys.exit(1)

if __name__ == "__main__":
    main()

def create_synthetic_dataset(vocab_size: int, num_samples: int = 500) -> Tuple[np.ndarray, np.ndarray]:

    X = np.random.randint(4, vocab_size, size=(num_samples,))  # Start from index 4 to reserve special tokens
    Y = np.zeros((num_samples, vocab_size), dtype=np.float32)
    for i in range(num_samples):
        target_id = np.random.randint(4, vocab_size)
        Y[i, target_id] = 1.0
    return X, Y

def load_real_dataset(file_path: str, vocab_size: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    tokens = word_tokenize(text.lower())
    freq = defaultdict(int)
    for token in tokens:
        freq[token] += 1

    # Add special tokens
    special_tokens = ["<PAD>", "<START>", "<END>", "<UNK>"]
    sorted_tokens = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    # Reserve indices for special tokens
    token_to_id = {token: idx for idx, token in enumerate(special_tokens)}
    for token, _ in sorted_tokens:
        if len(token_to_id) >= vocab_size:
            break
        if token not in token_to_id:
            token_to_id[token] = len(token_to_id)
    id_to_token = {idx: token for token, idx in token_to_id.items()}

    X, Y_ids = [], []
    for i in range(len(tokens)-1):
        current_token = tokens[i]
        next_token = tokens[i+1]
        X.append(token_to_id.get(current_token, token_to_id["<UNK>"]))
        Y_ids.append(token_to_id.get(next_token, token_to_id["<UNK>"]))

    Y = np.zeros((len(Y_ids), vocab_size), dtype=np.float32)
    for i, target_id in enumerate(Y_ids):
        Y[i, target_id] = 1.0

    return np.array(X), Y, token_to_id

def run_inference_without_examples():
  
    pass  

    main()
