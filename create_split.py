from pykeen.triples import TriplesFactory

tf = TriplesFactory.from_path("datasets/data_preprocessed/erias/graph_base.txt")
training, dev, testing = tf.split([0.8, 0.1, 0.1])

with open("datasets/data_preprocessed/erias/train.txt", "w") as out:
    for triple in training.mapped_triples.numpy():
        e1 = tf.entity_id_to_label[triple[0]] + "\t"
        r = tf.relation_id_to_label[triple[1]] + "\t"
        e2 = tf.entity_id_to_label[triple[2]] + "\n"
        literal = e1 + r + e2
        out.write(literal)

with open("datasets/data_preprocessed/erias/dev.txt", "w") as out:
    for triple in dev.mapped_triples.numpy():
        e1 = tf.entity_id_to_label[triple[0]] + "\t"
        r = tf.relation_id_to_label[triple[1]] + "\t"
        e2 = tf.entity_id_to_label[triple[2]] + "\n"
        literal = e1 + r + e2
        out.write(literal)

with open("datasets/data_preprocessed/erias/test.txt", "w") as out:
    for triple in testing.mapped_triples.numpy():
        e1 = tf.entity_id_to_label[triple[0]] + "\t"
        r = tf.relation_id_to_label[triple[1]] + "\t"
        e2 = tf.entity_id_to_label[triple[2]] + "\n"
        literal = e1 + r + e2
        out.write(literal)