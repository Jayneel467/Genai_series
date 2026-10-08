import ast
import importlib
import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"


class NotebookStructureTests(unittest.TestCase):
    def test_each_notebook_has_a_learning_guide(self):
        notebooks = sorted(NOTEBOOKS_DIR.glob("*.ipynb"))
        self.assertEqual(len(notebooks), 16)

        for notebook_path in notebooks:
            with self.subTest(notebook=notebook_path.name):
                notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
                self.assertTrue(notebook["cells"])
                first_cell = notebook["cells"][0]
                self.assertEqual(first_cell["cell_type"], "markdown")
                guide = "".join(first_cell["source"])
                self.assertTrue(guide.startswith("# "))
                self.assertIn("**Course author:** Jayneel Girnara", guide)
                headings = [
                    "".join(cell["source"]).strip().splitlines()[0]
                    for cell in notebook["cells"]
                    if cell["cell_type"] == "markdown"
                ]
                self.assertEqual(len(headings), len(set(headings)))
                if notebook_path.name != "00_course_foundations.ipynb":
                    lesson_text = "\n".join(
                        "".join(cell["source"])
                        for cell in notebook["cells"]
                        if cell["cell_type"] == "markdown"
                    )
                    self.assertIn("**Checkpoint:**", lesson_text)
                self.assertTrue(
                    notebook["metadata"]["language_info"]["version"].startswith(
                        "3.12"
                    )
                )
                self.assertEqual(notebook["metadata"]["kernelspec"]["name"], "python3")

    def test_code_cells_parse_as_python(self):
        for notebook_path in sorted(NOTEBOOKS_DIR.glob("*.ipynb")):
            notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
            for cell_number, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] != "code":
                    continue

                with self.subTest(
                    notebook=notebook_path.name,
                    cell=cell_number,
                ):
                    ast.parse("".join(cell["source"]))

    def test_readme_notebook_links_resolve(self):
        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("**Course author:** Jayneel Girnara", readme)
        for notebook_path in sorted(NOTEBOOKS_DIR.glob("*.ipynb")):
            self.assertIn(
                f"({notebook_path.relative_to(PROJECT_ROOT).as_posix()})",
                readme,
            )

    def test_foundations_lesson_links_resolve(self):
        foundations = (NOTEBOOKS_DIR / "00_course_foundations.ipynb").read_text(
            encoding="utf-8"
        )
        self.assertIn("large language model", foundations.lower())
        self.assertIn("embedding", foundations.lower())
        self.assertIn("Lesson 12", foundations)
        for notebook_path in sorted(NOTEBOOKS_DIR.glob("*.ipynb")):
            if notebook_path.name == "00_course_foundations.ipynb":
                continue
            self.assertIn(notebook_path.name, foundations)

    def test_direct_dependencies_are_pinned(self):
        requirements = (PROJECT_ROOT / "requirements.txt").read_text(
            encoding="utf-8"
        )
        dependencies = [
            line.strip()
            for line in requirements.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        self.assertTrue(dependencies)
        self.assertTrue(all("==" in dependency for dependency in dependencies))

    def test_provider_and_agent_integrations_import(self):
        modules = [
            "langchain_openai",
            "langchain_google_genai",
            "langchain_anthropic",
            "langchain_ollama",
            "langchain_groq",
            "langchain_community.tools.google_serper.tool",
            "langgraph.checkpoint.memory",
        ]
        for module in modules:
            with self.subTest(module=module):
                importlib.import_module(module)

    def test_local_document_loader_examples(self):
        from langchain_community.document_loaders import PyPDFLoader, TextLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        source_files = [
            (TextLoader, PROJECT_ROOT / "data" / "speech.txt"),
            (PyPDFLoader, PROJECT_ROOT / "data" / "data_science_syllabus.pdf"),
        ]
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )

        for loader_class, source_path in source_files:
            with self.subTest(source=source_path.name):
                documents = loader_class(source_path).load()
                self.assertTrue(documents)
                self.assertTrue(splitter.split_documents(documents))


if __name__ == "__main__":
    unittest.main()
