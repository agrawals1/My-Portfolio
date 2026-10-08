import numpy as np

from persona.embed import HashingEmbedder, PersonaIndex


def test_embeddings_are_unit_norm_and_deterministic():
    e = HashingEmbedder()
    a, b = e.embed(["Chief Financial Officer", "chief financial officer"])
    assert np.allclose(a, b) and abs(np.linalg.norm(a) - 1) < 1e-5
    assert (e.embed([""])[0] == 0).all()               # empty text -> zero vector, no NaN


def test_related_title_beats_unrelated_with_wide_margin(personas):
    ix = PersonaIndex(personas, HashingEmbedder())
    fin, odd = ix.scores(["chief financial officer", "chief happiness officer"])
    names = [p.persona_id for p in personas]
    assert names[int(fin.argmax())] == "P_FIN" and fin.max() > 0.35 > odd.max() + 0.2
