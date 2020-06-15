from collections import defaultdict
from io import StringIO
from string import punctuation

from nltk.tokenize import ToktokTokenizer
from pymorphy2 import MorphAnalyzer
from udapi.block.read.conllu import Conllu
from ufal.udpipe import Model, Pipeline


PUNCTUATION = punctuation + '«»—'
TOKENIZER = ToktokTokenizer()
MORPH = MorphAnalyzer()
SYNTAX_PARSER = Model.load("udpipe/russian-syntagrus.udpipe")
UDPIPE = Pipeline(SYNTAX_PARSER, "tokenize", Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')


def tokenize(text):
    return [token for token in TOKENIZER.tokenize(text) if token not in PUNCTUATION]


def _get_groups_from_tree(sentence):
    groups = defaultdict(list)

    to_tree = UDPIPE.process(sentence)
    tree = Conllu(filehandle=StringIO(to_tree)).read_tree()
    nodes = tree.descendants

    for node in nodes:
        parent = node.parent
        parent_form = parent.form
        parent_tag = parent.upos
        _node = node.form
        if parent_tag in ['NOUN', 'PROPN', 'VERB']:
            groups[(parent_form, parent_tag)].append((_node, node.upos))
    return groups


def _extract_vps(item, key):
    return [item[0], key[0]] if item[1] == 'ADV' else None


def _extract_nps(item, key):
    if item[1] == 'ADJ':
        return [item[0], key[0]]
    elif item[1] in ['NOUN', 'PROPN']:
        return [key[0], item[0]]
    else:
        return None


def _generate_relevant_groups(groups):
    extractors = {'PROPN': _extract_nps, 'NOUN': _extract_nps, 'VERB': _extract_vps}
    for group, children in groups.items():
        extractor = extractors[group[1]]
        for item in children:
            phrase = extractor(item, group)
            if phrase:
                yield phrase


def get_relevant_groups(sentence):
    relevant_groups = _generate_relevant_groups(_get_groups_from_tree(sentence))
    return [' '.join(MORPH.parse(word)[0].normal_form for word in group) for group in relevant_groups]
