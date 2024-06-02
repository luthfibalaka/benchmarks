import os


class CsvDataSource:
    def __init__(self, data_source: str):
        """
        Initialize the representation of a csv data source

        ### Parameters:
        - data_source (str): path to the data source with CSV files inside.
        """
        self._data_source = data_source
        self.csv_file_names = [
            f for f in os.listdir(self._data_source) if f.endswith(".csv")
        ]

    def __iter__(self):
        self.pointer = 0
        return self

    def __next__(self):
        """
        Return the file name and the contents.
        """
        if self.pointer >= len(self.csv_file_names):
            raise StopIteration

        csv_file_name = f"{self._data_source}/{self.csv_file_names[self.pointer]}"
        with open(csv_file_name) as csv_file:
            rows = csv_file.readlines()
            content = self._annotate_rows(rows)
            self.pointer += 1
            return (csv_file_name, content)

    def _annotate_rows(self, rows):
        """
        Annotate each row with "col: " if it is a column row
        or "row i" if it is a data row.

        ### Parameters:
        - rows (list[str])

        ### Returns:
        - annotated_rows (list[str])
        """
        annotated_rows = []
        for i in range(len(rows)):
            if i == 0:
                annotated_rows.append("col: " + rows[i].lstrip())
            elif i == len(rows) - 1:
                annotated_rows.append(f"row {i}: " + rows[i].strip())
            else:
                annotated_rows.append(f"row {i}: " + rows[i].lstrip())
        return annotated_rows

    def set_data_source(self, data_source: str):
        """
        Re-assign the data source
        """
        self._data_source = data_source
        self.csv_file_names = [
            f for f in os.listdir(self._data_source) if f.endswith(".csv")
        ]
