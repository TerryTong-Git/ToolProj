import ast
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Type

from pydantic import BaseModel, Field

from src.exps_performance.problems.nphardeval import NpCheckAndFormat, NpQuestion

msp_desc = (
    "Description: The meeting scheduling problem (MSP) is a type of constraint satisfaction problem where the goal is to find a suitable time slot for a meeting that all participants can attend without conflicts in their schedules."
    "Question: There are {total_participants} participants with their available time slots. There are {total_timeslots} consecutive non-overlapping time slots. Let's assume all meetings has duration of 1."
)

func_typing = "Dict[int, List[int]]"  # (Meeting #Num -> Time slots)


class MspAnswer(BaseModel):
    Meet2Time: str = Field(
        description="The meeting number to list of time slots. Type: Dict[int, List[int]]. For example: '{{0:[1,2], 1:[4], ...}}' ", default=""
    )


@dataclass
class MspQuestion(NpQuestion):
    kind: str = "msp"
    type: str = "code"  # could be sim, nl etc
    time_slots: int = -1  # type: ignore
    meetings: List[Dict[str, int]] = field(default_factory=list)
    participants: Dict[int, Dict[str, List[int]]] = field(default_factory=dict)
    complexity_level: int = -1
    code: str = ""

    @property
    def util_pointer(self) -> Type[NpCheckAndFormat]:
        return MspCheckAndFormat


class MspCheckAndFormat(NpCheckAndFormat):
    def __init__(self, prob_type: str):
        super().__init__(prob_type, func_typing, msp_desc, MspAnswer)
        self.instancetype = MspQuestion

    def loaded_data_to_class(self, data: Any) -> Any:
        return data

    def type_check_code(self, code: str) -> bool:
        try:
            evaluated = ast.literal_eval(str(code))
        except (SyntaxError, ValueError):
            return False  # f"Syntax or Value Error {e}"
        if not isinstance(evaluated, dict):
            return False  # "Not a dict"
        else:
            for m_id, time_slots in evaluated.items():
                if not isinstance(m_id, int):
                    return False  # "M_id wrong type"
                if not isinstance(time_slots, list):
                    return False  # "Time slots not a list"
                for time in time_slots:
                    if not isinstance(time, int):
                        return False  # "Time is not an int"
        return True

    # tied to code
    def get_field_kwargs(self, result: Any) -> dict[str, str]:
        return dict(Meet2Time=str(result))

    @property  # should be an abstract property implemented by all classes to decide which template to use
    def prompt(self) -> Any:
        return (
            self.prompt_template(["total_participants", "total_timeslots", "meetingdetails"])
            if self.prob_type != "sim"
            else self.prompt_template(["code"])
        )

    def format_one(self, q: MspQuestion) -> str:
        if self.prob_type == "sim":
            return str(self.prompt.format_prompt(code=q.code).to_string())
        participants = q.participants
        meetingdetails = "\n The meetings and participants details are as below: \n"
        for meeting in q.meetings:
            this_line = "Meeting {} is with duration {}.".format(meeting["id"], meeting["duration"])
            meetingdetails += this_line + "\n"
        for j in participants.keys():
            this_line = "Participant {} is available at time slots {} and has meetings {}.".format(
                j, participants[j]["available_slots"], participants[j]["meetings"]
            )
            meetingdetails += this_line + "\n"
        prompt_text = self.prompt.format_prompt(total_participants=participants, total_timeslots=q.time_slots, meetingdetails=meetingdetails)
        return str(prompt_text.to_string())

    def decision_check(self, q: MspQuestion, output: BaseModel) -> tuple[bool, str]:
        """
        Validate the MSP solution.

        Parameters:
        - q: The MSP q as a dictionary.
        - solution: A dictionary with meeting ids as keys and lists of scheduled time slots as values.

        Returns:
        - A tuple (is_valid, message). is_valid is True if the solution is valid, False otherwise.
        message contains information about the validity of the solution.
        """
        try:
            output_dict = ast.literal_eval(str(output.Meet2Time))
        except (SyntaxError, TypeError):
            return False, "Parse error"
        if not self.type_check_code(output.Meet2Time) or output_dict is None:
            return False, "Parse error"
        # Check if all meetings are scheduled within the available time slots
        for meeting in q.meetings:
            m_id = meeting["id"]
            duration = meeting["duration"]
            scheduled_slots = output_dict.get(m_id, None)

            # Check if the meeting is scheduled
            if scheduled_slots is None:
                return False, f"Meeting {m_id} is not scheduled."

            # Check if the meeting fits within the number of total time slots
            if any(slot >= q.time_slots for slot in scheduled_slots):
                return False, f"Meeting {m_id} does not fit within the available time slots."

            # Check if the scheduled slots are contiguous and fit the meeting duration
            if len(scheduled_slots) != duration or not all(scheduled_slots[i] + 1 == scheduled_slots[i + 1] for i in range(len(scheduled_slots) - 1)):
                return False, f"Meeting {m_id} is not scheduled in contiguous time slots fitting its duration."

            # Check if all participants are available at the scheduled time
            for p_id, participant in q.participants.items():
                if m_id in participant["meetings"]:
                    if not all(slot in participant["available_slots"] for slot in scheduled_slots):
                        return False, f"Participant {p_id} is not available for meeting {m_id} at the scheduled time."

        # Check if any participant is double-booked
        participants_schedule: Dict[int, List] = {p_id: [] for p_id in q.participants}
        for m_id, time_slots in output_dict.items():
            try:
                duration = next(meeting["duration"] for meeting in q.meetings if meeting["id"] == m_id)
                if len(time_slots) != duration:
                    return False, f"Meeting {m_id} duration does not match the number of scheduled time slots."
                for p_id, participant in q.participants.items():
                    if m_id in participant["meetings"]:
                        participants_schedule[p_id].extend(time_slots)
            except:  # noqa
                return False, f"Meeting {m_id} is not in the instance or program error."

        for p_id, slots in participants_schedule.items():
            if len(slots) != len(set(slots)):
                return False, f"Participant {p_id} is double-booked."

        return True, "The output is valid."

    def load_data(self) -> list[MspQuestion]:
        with open(os.path.join(self.folder_name, "MSP", "msp_instances.json"), "r") as f:
            data = json.load(f)
        problem = self.instancetype  # type: ignore
        data_func = self.loaded_data_to_class  # type: ignore #for some reason can only see base class type...
        all_data = [problem(**data_func(d)) for d in data]
        return list(all_data)
