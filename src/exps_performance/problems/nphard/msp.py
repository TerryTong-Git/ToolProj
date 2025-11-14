import ast
import json
from typing import Any

from src.exps_performance.problems.nphardeval import NPHardEvalProblem
from src.exps_performance.problems.prompts import mspPrompts


class MSP(NPHardEvalProblem):
    def __init__(self):
        self.p = mspPrompts

    def format_one(self, q: Any) -> str:
        participants = q["participants"]
        prompt_text = (
            self.instantiate_prompt(dict(total_participants=participants, total_timeslots=q["time_slots"]))
            + "\n The meetings and participants details are as below: \n"
        )

        for meeting in q["meetings"]:
            this_line = "Meeting {} is with duration {}.".format(meeting["id"], meeting["duration"])
            prompt_text += this_line + "\n"
        for j in participants.keys():
            this_line = "Participant {} is available at time slots {} and has meetings {}.".format(
                j, participants[j]["available_slots"], participants[j]["meetings"]
            )
            prompt_text += this_line + "\n"
        return prompt_text

    def decision_check(self, q, output):
        """
        Validate the MSP solution.

        Parameters:
        - q: The MSP q as a dictionary.
        - solution: A dictionary with meeting ids as keys and lists of scheduled time slots as values.

        Returns:
        - A tuple (is_valid, message). is_valid is True if the solution is valid, False otherwise.
        message contains information about the validity of the solution.
        """
        # print(llm_string)
        # print(output.text)

        # convert output to dictionary
        if output == "":
            return False
        elif output is None:
            return False
        else:
            if isinstance(output, str):
                try:
                    output = ast.literal_eval(output)
                    if output is None:
                        return False
                except:  # noqa
                    try:
                        output = ast.literal_eval("{" + output + "}")
                        if output is None:
                            return False
                    except:  # noqa
                        return False
            else:
                try:
                    output = ast.literal_eval(output.text)
                    if output is None:
                        return False
                except:  # noqa
                    return False
        # convert key type to int
        if isinstance(output, dict):
            print(output)
            output = {int(k): v for k, v in output.items()}
        else:
            return False

        # Check if all meetings are scheduled within the available time slots
        for meeting in q["meetings"]:
            m_id = meeting["id"]
            duration = meeting["duration"]
            scheduled_slots = output.get(m_id, None)

            # Check if the meeting is scheduled
            if scheduled_slots is None:
                return False, f"Meeting {m_id} is not scheduled."

            # Check if the meeting fits within the number of total time slots
            if any(slot >= q["time_slots"] for slot in scheduled_slots):
                return False, f"Meeting {m_id} does not fit within the available time slots."

            # Check if the scheduled slots are contiguous and fit the meeting duration
            if len(scheduled_slots) != duration or not all(scheduled_slots[i] + 1 == scheduled_slots[i + 1] for i in range(len(scheduled_slots) - 1)):
                return False, f"Meeting {m_id} is not scheduled in contiguous time slots fitting its duration."

            # Check if all participants are available at the scheduled time
            for p_id, participant in q["participants"].items():
                if m_id in participant["meetings"]:
                    if not all(slot in participant["available_slots"] for slot in scheduled_slots):
                        return False, f"Participant {p_id} is not available for meeting {m_id} at the scheduled time."

        # Check if any participant is double-booked
        participants_schedule = {p_id: [] for p_id in q["participants"]}
        for m_id, time_slots in output.items():
            try:
                duration = next(meeting["duration"] for meeting in q["meetings"] if meeting["id"] == m_id)
                if len(time_slots) != duration:
                    return False, f"Meeting {m_id} duration does not match the number of scheduled time slots."
                for p_id, participant in q["participants"].items():
                    if m_id in participant["meetings"]:
                        participants_schedule[p_id].extend(time_slots)
            except:  # noqa
                return False, f"Meeting {m_id} is not in the instance or program error."

        for p_id, slots in participants_schedule.items():
            if len(slots) != len(set(slots)):
                return False, f"Participant {p_id} is double-booked."

        return True, "The output is valid."

    @staticmethod
    def load_data(data_path):
        with open(data_path + "msp_instances.json", "r") as f:
            all_data = json.load(f)
        return all_data
