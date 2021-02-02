"""Script to convert MultiWOZ 2.2 from SGD format to MultiWOZ format."""
import glob
import json
import os

from absl import app
from absl import flags
from absl import logging

FLAGS = flags.FLAGS

flags.DEFINE_string("multiwoz21_data_dir", None,
                    "Path of the MultiWOZ 2.1 dataset.")
flags.DEFINE_string("output_file", None, "Output file path in MultiWOZ format.")

_UNTRACKED_SLOTS = frozenset({
    "taxi-bookphone", "train-booktrainid", "taxi-booktype",
    "restaurant-bookreference", "hospital-bookreference", "hotel-bookreference",
    "train-bookreference", "hospital-booktime"
})
_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
flags.mark_flags_as_required(["multiwoz21_data_dir", "output_file"])


def get_slot_name(slot_name, service_name, in_book_field=False):
  """Get the slot name that is consistent with the schema file."""
  slot_name = ("book" + slot_name if in_book_field and slot_name != "department"
               and slot_name != "name" else slot_name)
  return "-".join([service_name, slot_name]).lower()


def format_states(groundtruth_states, states_to_correct):
  """Correct the dialogue states in place."""
  for domain_name, values in states_to_correct.items():
    for k, v in values["book"].items():
      if isinstance(v, list):
        for item_dict in v:
          for slot_name in item_dict:
            new_slot_name = get_slot_name(
                slot_name, domain_name, in_book_field=True)
            if new_slot_name in _UNTRACKED_SLOTS:
              continue
            # For the tracked slots, correct their states.
            if new_slot_name in groundtruth_states:
              item_dict[slot_name] = groundtruth_states[new_slot_name]
            else:
              item_dict[slot_name] = []
      if isinstance(v, str):
        slot_name = get_slot_name(k, domain_name, in_book_field=True)
        if slot_name in _UNTRACKED_SLOTS:
          continue
        if slot_name in groundtruth_states:
          values["book"][k] = groundtruth_states[slot_name]
        else:
          values["book"][k] = []

    for slot_name in values["semi"]:
      new_slot_name = get_slot_name(slot_name, domain_name)
      # All the slots in "semi" are tracked.
      if new_slot_name in groundtruth_states:
        values["semi"][slot_name] = groundtruth_states[new_slot_name]
      else:
        values["semi"][slot_name] = []


def main(argv):
  data_path = os.path.join(FLAGS.multiwoz21_data_dir, "data.json")
  with open(data_path, "r") as f:
    multiwoz_data = json.load(f)
  file_pattern = os.path.join(_DIR_PATH, "*/dialogues_*.json")
  files = glob.glob(file_pattern)
  clean_data = {}
  for file_name in files:
    with open(file_name, "r") as f:
      dialogues = json.load(f)
      for dialogue in dialogues:
        clean_data[dialogue["dialogue_id"]] = dialogue
  # Load action file.
  action_file = os.path.join(_DIR_PATH, "dialog_acts.json")
  with open(action_file, "r") as f:
    action_data = json.load(f)
  dialogue_ids = list(multiwoz_data.keys())
  for dialogue_id in dialogue_ids:
    dialogue = multiwoz_data[dialogue_id]["log"]
    if dialogue_id not in clean_data:
      logging.info("Dialogue %s doesn't exist in MultiWOZ 2.2.", dialogue_id)
      del multiwoz_data[dialogue_id]
      continue
    clean_dialogue = clean_data[dialogue_id]
    for i, turn in enumerate(dialogue):
      # Update the utterance.
      turn["text"] = clean_dialogue["turns"][i]["utterance"]
      dialog_act = {}
      span_info = []
      if str(i) in action_data[dialogue_id]:
        dialog_act = action_data[dialogue_id][str(i)]["dialog_act"]
        span_info = action_data[dialogue_id][str(i)]["span_info"]
      turn["dialog_act"] = dialog_act
      turn["span_info"] = span_info
      # Skip user turns because states are written in the system turns.
      if i % 2 == 0:
        continue
      clean_states = {}
      for frame in clean_dialogue["turns"][i - 1]["frames"]:
        clean_states.update(frame["state"]["slot_values"])
      format_states(clean_states, turn["metadata"])
  with open(FLAGS.output_file, "w") as f:
    json.dump(multiwoz_data, f, indent=2, separators=(",", ": "), sort_keys=True)
  logging.info("Finish writing %d dialogues", len(multiwoz_data))


if __name__ == "__main__":
  app.run(main)
