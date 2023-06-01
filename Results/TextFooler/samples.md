Samples.pkl contains serialized and structured predictions
that were successfully manipulated by the model.

samples = [(fromLabel, toLabel, x_orig, x_adv)]
    fromLabel   : Ground-truth label
    toLabel     : Predicted label
    x_orig      : Original text
    x_adv       : Adversarial text

labels
    0 : "SUPPORTS"
    1 : "NOT ENOUGH INFORMATION"
    2 : "REFUTES"

Weird examples 26; 30; 31; 36
