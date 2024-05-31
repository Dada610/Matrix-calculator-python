# matrix_calculator.py
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label
from kivy.uix.spinner import Spinner
from kivy.uix.scrollview import ScrollView
from kivy.uix.popup import Popup
import numpy as np

class MatrixCalculatorApp(App):
    def build(self):
        self.operators = [
            "Addition", "Subtraction", "Multiplication", "Determinant", "Inverse", "Transpose",
            "Rank", "Multiply by Scalar", "Row Echelon Form", "Diagonal Matrix", "To the Power of",
            "LU Decomposition", "Cholesky Decomposition", "Solve Linear System"
        ]
        self.history = []

        layout = BoxLayout(orientation='vertical', padding=10)

        self.result_label = Label(text="Result: ", font_size=20)
        layout.add_widget(self.result_label)

        self.matrix_input_a = TextInput(hint_text='Enter Matrix A', multiline=True)
        layout.add_widget(self.matrix_input_a)

        self.matrix_input_b = TextInput(hint_text='Enter Matrix B', multiline=True)
        layout.add_widget(self.matrix_input_b)

        self.operator_spinner = Spinner(
            text="Choose Operation",
            values=self.operators,
            on_text=self.update_spinner_text
        )
        layout.add_widget(self.operator_spinner)

        self.calculate_button = Button(text="Calculate", on_press=self.calculate)
        layout.add_widget(self.calculate_button)

        self.clear_history_button = Button(text="Clear History", on_press=self.clear_history)
        layout.add_widget(self.clear_history_button)

        self.save_history_button = Button(text="Save History", on_press=self.save_history)
        layout.add_widget(self.save_history_button)

        self.load_history_button = Button(text="Load History", on_press=self.load_history)
        layout.add_widget(self.load_history_button)

        self.history_label = Label(text="History:", font_size=16)
        layout.add_widget(self.history_label)

        self.history_scrollview = ScrollView()
        self.history_layout = BoxLayout(orientation='vertical', spacing=5, size_hint_y=None)
        self.history_scrollview.add_widget(self.history_layout)
        layout.add_widget(self.history_scrollview)

        return layout

    def update_spinner_text(self, spinner, text):
        spinner.text = text

    def calculate(self, instance):
        matrix_a = self.parse_matrix(self.matrix_input_a.text)
        matrix_b = self.parse_matrix(self.matrix_input_b.text)

        if matrix_a is None:
            self.result_label.text = "Invalid matrix input"
            return

        operation = self.operator_spinner.text
        result_matrix = self.perform_operation(matrix_a, matrix_b, operation)

        if result_matrix is not None:
            self.result_label.text = f"Result: {result_matrix}"
            self.add_to_history(f"{operation}: {result_matrix}")
            self.clear_matrix_inputs()
        else:
            self.result_label.text = "Error in calculation"

    def parse_matrix(self, input_text):
        try:
            rows = input_text.strip().split('\n')
            matrix = [[float(entry) for entry in row.split()] for row in rows]
            return matrix
        except ValueError:
            return None

    def perform_operation(self, matrix_a, matrix_b, operation):
        if operation == "Addition":
            return np.add(matrix_a, matrix_b)
        elif operation == "Subtraction":
            return np.subtract(matrix_a, matrix_b)
        elif operation == "Multiplication":
            return np.dot(matrix_a, matrix_b)
        elif operation == "Determinant":
            return np.linalg.det(matrix_a)
        elif operation == "Inverse":
            return np.linalg.inv(matrix_a)
        elif operation == "Transpose":
            return np.transpose(matrix_a)
        elif operation == "Rank":
            return np.linalg.matrix_rank(matrix_a)
        elif operation == "Multiply by Scalar":
            scalar = float(input("Enter scalar value: "))
            return np.multiply(matrix_a, scalar)
        elif operation == "Row Echelon Form":
            return self.row_echelon_form(matrix_a)
        elif operation == "Diagonal Matrix":
            return np.diag(np.diag(matrix_a))
        elif operation == "To the Power of":
            power = int(input("Enter the power: "))
            return np.linalg.matrix_power(matrix_a, power)
        elif operation == "LU Decomposition":
            return self.lu_decomposition(matrix_a)
        elif operation == "Cholesky Decomposition":
            return self.cholesky_decomposition(matrix_a)
        elif operation == "Solve Linear System":
            return self.solve_linear_system(matrix_a)
        else:
            return None

    def row_echelon_form(self, matrix_a):
        augmented_matrix = np.array(matrix_a, dtype=float)
        num_rows, num_cols = augmented_matrix.shape

        for i in range(num_rows):
            # Find the first non-zero element in the current column
            nonzero_row = np.nonzero(augmented_matrix[i:, i])[0]
            if len(nonzero_row) == 0:
                continue  # Skip to the next column if no non-zero elements are found
            else:
                augmented_matrix[[i, i + nonzero_row[0]]] = augmented_matrix[[i + nonzero_row[0], i]]

            pivot_row = augmented_matrix[i, i:]
            pivot_element = pivot_row[0]

            # Normalize the pivot row
            augmented_matrix[i, i:] /= pivot_element

            # Eliminate other rows
            for j in range(i + 1, num_rows):
                factor = augmented_matrix[j, i]
                augmented_matrix[j, i:] -= factor * pivot_row

        return augmented_matrix

    def lu_decomposition(self, matrix_a):
        return np.array(lu(matrix_a), dtype=float)

    def cholesky_decomposition(self, matrix_a):
        try:
            return cholesky(matrix_a, lower=True)
        except np.linalg.LinAlgError:
            return "Cholesky decomposition is not applicable for this matrix."

    def solve_linear_system(self, augmented_matrix):
        num_equations = len(augmented_matrix)
        num_variables = len(augmented_matrix[0]) - 1
        solution = [0] * num_variables

        for i in range(num_equations):
            # Check for zero pivot
            if augmented_matrix[i][i] == 0:
                return "No unique solution. Zero pivot encountered."

            # Eliminate variables above the pivot
            for j in range(i):
                factor = augmented_matrix[j][i] / augmented_matrix[i][i]
                augmented_matrix[j] = [a - factor * b for a, b in zip(augmented_matrix[j], augmented_matrix[i])]

            # Eliminate variables below the pivot
            for j in range(i + 1, num_equations):
                factor = augmented_matrix[j][i] / augmented_matrix[i][i]
                augmented_matrix[j] = [a - factor * b for a, b in zip(augmented_matrix[j], augmented_matrix[i])]

        # Back substitution
        for i in range(num_equations - 1, -1, -1):
            solution[i] = (augmented_matrix[i][-1] - sum(
                augmented_matrix[i][j] * solution[j] for j in range(i + 1, num_variables))) / augmented_matrix[i][i]

        return solution

    def add_to_history(self, entry):
        self.history.append(entry)
        self.update_history_ui()

    def update_history_ui(self):
        self.history_layout.clear_widgets()
        for entry in self.history:
            label = Label(text=entry, font_size=14)
            self.history_layout.add_widget(label)

    def clear_history(self, instance):
        self.history = []
        self.update_history_ui()

    def clear_matrix_inputs(self):
        self.matrix_input_a.text = ""
        self.matrix_input_b.text = ""

    def save_history(self, instance):
        try:
            with open("history.txt", "w") as file:
                for entry in self.history:
                    file.write(f"{entry}\n")
            self.show_message("History saved successfully.")
        except Exception as e:
            self.show_message(f"Error saving history: {str(e)}")

    def load_history(self, instance):
        try:
            with open("history.txt", "r") as file:
                lines = file.readlines()
                self.history = [line.strip() for line in lines]
                self.update_history_ui()
            self.show_message("History loaded successfully.")
        except FileNotFoundError:
            self.show_message("History file not found.")
        except Exception as e:
            self.show_message(f"Error loading history: {str(e)}")

    def show_message(self, message):
        popup = BoxLayout(orientation='vertical', padding=10)
        popup_label = Label(text=message, font_size=16, size_hint_y=None)
        popup.add_widget(popup_label)
        popup_button = Button(text="OK", size_hint_y=None, height=40)
        popup_button.bind(on_press=lambda instance: popup_parent.remove_widget(popup))
        popup.add_widget(popup_button)

        popup_parent = BoxLayout()
        popup_parent.add_widget(popup)

        popup_window = Popup(title="Message", content=popup_parent, size_hint=(None, None), size=(300, 150))
        popup_window.open()

if __name__ == '__main__':
    MatrixCalculatorApp().run()
