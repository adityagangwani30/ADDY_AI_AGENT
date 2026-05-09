from services.document_qa import answer_question
from memory.file_index import FileIndex


def test_document_qa_simple():
    idx = FileIndex(db_path=":memory:")
    # add a small document
    idx.add_file(file_id="fqa1", filename="notes.txt", extracted_text="The Sales Sense model achieved 92% accuracy on test set.", keywords="sales,accuracy", source="local")
    answer = answer_question(user_id="u1", question="What accuracy did my Sales Sense model achieve?", request_id="t1")
    assert isinstance(answer, str)
