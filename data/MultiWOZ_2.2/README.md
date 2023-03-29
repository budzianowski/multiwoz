# MultiWOZ 2.2

This dataset consists of a schema file `schema.json` describing the ontology and
dialogue files `dialogues_*.json` of dialogue data under the `train`, `dev`, and
`test` folders.

**Notes:**

- Compared to MultiWOZ 2.1, we remove `SNG01862.json` as it's an invalid dialogue.
- MultiWOZ 2.2 is also available on [Hugging Face](https://huggingface.co/datasets/multi_woz_v22) and [ParlAI](https://parl.ai/docs/tasks.html). 

## Schema file

`schema.json` defines the new ontology using the schema representation in the
[schema-guided dialogue dataset](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue#scheme-representation]).

The table below shows the categorical slots, non-categorical slots and intents
defined for each domain.

| Domain     | Categorical slots       | Non-categorical slots   | Intents    |
| ---------- | :---------------------: | :---------------------: | :--------: |
| Restaurant | pricerange, area, bookday, bookpeople | food, name, booktime, address, phone, postcode, ref | find, book |
| Attraction | area, type              | name, address, entrancefee, openhours, entrancefee, openhours, phone, postcode    | find       |
| Hotel      | pricerange, parking, internet, stars, area, type, bookpeople, bookday, bookstay  | name, address, phone, postcode, ref | find, book |
| Taxi       | -                       | destination, departure, arriveby, leaveat, phone, type | book       |
| Train      | destination, departure, day, bookpeople | arriveby, leaveat, trainid, ref, price, duration | find, book |
| Bus        | day                     | departure, destination, leaveat | find       |
| Hospital   | -                       | department , address, phone, postcode | find       |
| Police     | -                       | name, address, phone, postcode | find       |

Of the 61 slots in the schema, the following 35 slots are tracked in the
dialogue state:

```
{
    "attraction-area",
    "attraction-name",
    "attraction-type",
    "bus-day",
    "bus-departure",
    "bus-destination",
    "bus-leaveat",
    "hospital-department",
    "hotel-area",
    "hotel-bookday",
    "hotel-bookpeople",
    "hotel-bookstay",
    "hotel-internet",
    "hotel-name",
    "hotel-parking",
    "hotel-pricerange",
    "hotel-stars",
    "hotel-type",
    "restaurant-area",
    "restaurant-bookday",
    "restaurant-bookpeople",
    "restaurant-booktime",
    "restaurant-food",
    "restaurant-name",
    "restaurant-pricerange",
    "taxi-arriveby",
    "taxi-departure",
    "taxi-destination",
    "taxi-leaveat",
    "train-arriveby",
    "train-bookpeople",
    "train-day",
    "train-departure",
    "train-destination",
    "train-leaveat"
}
```

## Dialogue files

Dialogues are formatted following the data presentation of the
[schema-guided dialogue dataset](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue#dialogue-representation).

**Because the state value of a slot can be mentioned in different ways in the
dialogues (e.g. 8pm and 20:00), the ground truth state values is presented as a
list of values to incorporate such cases. <span style="color:red">Predicting any
of them is considered as correct in the evaluation.</span>** Specifically, the
state values of each turn is represented as:

```
{
  "state":{
    "active_intent": String. User intent of the current turn.
    "requested_slots": List of string representing the slots, the values of which are being requested by the user.
    "slot_values": Dict of state values. The key is slot name in string. The value is a list of values.
  }
}
```

In addition, we also add the span annotations that identify the location where
slot values have been mentioned in the utterances for non-categorical slots.
These span annotations are represented as follows:

```
{
  "slots": [
    {
      "slot": String of slot name.
      "start": Int denoting the index of the starting character in the utterance corresponding to the slot value.
      "exclusive_end": Int denoting the index of the character just after the last character corresponding to the slot value in the utterance. In python, utterance[start:exclusive_end] gives the slot value.
      "value": String of value. It equals to utterance[start:exclusive_end], where utterance is the current utterance in string.
    }
  ]
}
```

There are some non-categorical slots whose values are carried over from another
slot in the dialogue state. Their values don"t explicitly appear in the
utterances.

For example, a user utterance can be *"I also need a taxi from the restaurant to
the hotel."*, in which the state values of *"taxi-departure"* and
*"taxi-destination"* are respectively carried over from that of
*"restaurant-name"* and *"hotel-name"*. **For these slots, instead of annotating
them as spans, we use a <span style="color:red">"copy from" annotation</span> to
identify the slot it copies the value from.** This annotation is formatted as
follows,

```
{
  "slots": [
    {
      "slot": Slot name string.
      "copy_from": The slot to copy from.
      "value": A list of slot values being . It corresponds to the state values of the "copy_from" slot.
    }
  ]
}
```

## Action annotation

There are 8,333 turns missing dialogue action annotations in MultiWOZ 2.1. We
used a finetuned [T5 model](https://github.com/google-research/text-to-text-transfer-transformer) to annotate actions for these missing turns, and manually
verified and corrected them. Please note that there are still 749
turns without dialogue action annotations because the semantics of the
utterances can"t be appropriately expressed using
[the dialogue actions defined by ConvLab](https://github.com/ConvLab/ConvLab/blob/master/data/multiwoz/annotation/Multiwoz%20data%20analysis.md#dialog-act),
such as *"Sure. Just a moment."*, *"said to skip."*, etc.

Please check the annotated action annotation in "dialog_acts.json". It is
formatted in the same style as MultiWOZ 2.1 except that we use character-level
indexing instead of token-level indexing for the action values.

```
{
  "$dialogue_id": [
    "$turn_id": {
      "dialogue_acts": {
        "$act_name": [
          [
            "$slot_name",
            "$action_value"
          ]
        ]
      },
      "span_info": [
        [
          "$act_name"
          "$slot_name",
          "$action_value"
          "$start_charater_index",
          "$exclusive_end_character_index"
        ]
      ]
    }
  ]
}
```

## Conversion to the data format of MultiWOZ 2.1

To include the corrections from MultiWOZ 2.2 dataset into MultiWOZ 2.1 in the
format used by the MultiWOZ 2.1 dataset, please download the
[MultiWOZ 2.1](https://github.com/budzianowski/multiwoz/blob/master/data/MultiWOZ_2.1.zip)
zip file, unzip it, and run

```bash
python convert_to_multiwoz_format.py --multiwoz21_data_dir=<multiwoz21_data_dir> --output_file=<output json file>
```

Please refer to our
[paper](https://www.aclweb.org/anthology/2020.nlp4convai-1.13.pdf) for more
details about the dataset.


## Questions

We are continuously making efforts to make this dataset better. If you have any
questions, please feel free to contact us by (schema-guided-dst@google.com or
xiaoxuez@google.com).

