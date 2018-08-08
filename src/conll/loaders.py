# Loader functions for Conll Data
import codecs
import re

DOCSTART_MARKER = '-DOCSTART-'
RE_WIKI_ENT = re.compile(r'.*wiki\/(.*)')


def is_training_doc(doc_id):
    return 'test' not in doc_id


def is_test_doc(doc_id):
    return 'testb' in doc_id


def is_dev_doc(doc_id):
    return 'testa' in doc_id


def doc_tag_for_id(doc_id):
    if 'testa' in doc_id:
        return 'dev'
    elif 'testb' in doc_id:
        return 'test'
    return 'train'


def iter_docs(path, doc_id_predicate, redirects=None):
    redirects = redirects or {}
    with codecs.open(path, 'r', 'utf-8') as f:
        doc_id = None
        doc_tokens = None
        doc_mentions = None
        doc_count = 0
        num_tokens = 0
        proper_mentions = []
        mention_no = 0
        for line in f:
            parts = line.split('\t')
            if len(parts) > 0:
                token = parts[0].strip()
                if not token.startswith(DOCSTART_MARKER):
                    num_tokens += 1
                # if this line contains a mention
                if len(parts) >= 4 and parts[1] == 'B':
                    # filter empty and non-links
                    mention_no += 1
                    if parts[3].strip() != '' and not parts[3].startswith('--'):
                        proper_mentions.append(mention_no)
                        try:
                            entity = RE_WIKI_ENT.match(parts[4]).group(1)
                        except AttributeError:
                            print(parts[4])
                        entity = redirects.get(entity, entity)
                        begin = sum(len(t)+1 for t in doc_tokens)

                        dodgy_tokenisation_bs_offset = 1 if re.search('[A-Za-z],',parts[2]) else 0

                        position = (begin, begin + len(parts[2]) + dodgy_tokenisation_bs_offset)
                        doc_mentions.append((entity, position))

                if token.startswith(DOCSTART_MARKER):
                    if doc_id is not None and doc_id_predicate(doc_id):
                        doc_count += 1
                        yield (' '.join(doc_tokens),doc_mentions, num_tokens, proper_mentions, doc_id)
                        num_tokens = 0

                    doc_id = token[len(DOCSTART_MARKER) + 2:-1]

                    ## TODO: FIX THIS HACK
                    if doc_id[:3] == '618' and len(doc_tokens) == 510:
                        yield (' '.join(doc_tokens),doc_mentions, num_tokens, proper_mentions, doc_id)
                    doc_tokens = []
                    doc_mentions = []
                elif doc_id is not None:
                    doc_tokens.append(token)

        if doc_id is not None and doc_id_predicate(doc_id):
            yield (' '.join(doc_tokens), doc_mentions, num_tokens, proper_mentions, doc_id)


def tag(doc_text, nlp=None):
    spacy_doc = nlp(doc_text)
    ents = [ent.text for ent in spacy_doc.ents]
    poss = [(ent.start_char, ent.end_char) for ent in spacy_doc.ents]
    return ents, poss