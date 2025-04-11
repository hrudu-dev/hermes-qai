import cirq
import numpy as np


class QuantumEmbedding:
    def __init__(self, num_qubits=4):
        self.num_qubits = num_qubits
        self.qubits = cirq.LineQubit.range(num_qubits)
        self.circuit = cirq.Circuit()
        self.params = [cirq.Symbol(f'theta_{i}') for i in range(num_qubits)]
        self._build_circuit()

    def _build_circuit(self):
        # Encode classical data using Ry rotations
        for q, theta in zip(self.qubits, self.params):
            self.circuit.append(cirq.ry(theta)(q))

        # Add entanglement
        for i in range(self.num_qubits - 1):
            self.circuit.append(cirq.CNOT(self.qubits[i], self.qubits[i + 1]))

        # Measure all qubits
        self.circuit.append(cirq.measure(*self.qubits, key='result'))

    def encode(self, classical_input):
        if len(classical_input) != self.num_qubits:
            raise ValueError(f"Expected input of length {self.num_qubits}, got {len(classical_input)}")

        resolver = cirq.ParamResolver({
            f'theta_{i}': val for i, val in enumerate(classical_input)
        })

        simulator = cirq.Simulator()
        result = simulator.run(self.circuit, resolver, repetitions=1)
        return np.array([int(bit) for bit in result.measurements['result'][0]])


# 🔍 Example usage
if __name__ == "__main__":
    q_embed = QuantumEmbedding(num_qubits=4)
    input_data = [np.pi / 2, np.pi / 4, np.pi / 6, np.pi / 8]
    q_output = q_embed.encode(input_data)
    print("Quantum output:", q_output)
