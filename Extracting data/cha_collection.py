import os
from typing import Iterator, Callable
from collection import Collection, RawDataPoint, NormalizedDataPoint
import csv
import json
from dataclasses import asdict


class CHACollection(Collection):
    def __init__(self, path: str, language: str, override_diagnosis: str = None,
                 subfolder_label_map: dict = None, filename_label_map: dict = None):
        super().__init__(path, language)
        self.override_diagnosis = override_diagnosis
        self.subfolder_label_map = subfolder_label_map
        self.filename_label_map = filename_label_map

    def _get_parser(self) -> Callable[[dict, str], None]:
        if self.language in ("english", "chinese", "korean", "german", "greek"):
            return self._parse_line_general
        elif self.language == "spanish":
            return self._parse_line_spanish
        else:
            raise ValueError(f"Unsupported language: {self.language}")

    def _infer_label_from_filename(self, filename: str):
        """Match filename against filename_label_map prefixes."""
        if not self.filename_label_map:
            return None
        basename = os.path.splitext(filename)[0]
        for prefix, label in self.filename_label_map.items():
            if basename.startswith(prefix):
                return label
        return None

    def _collect_cha_files(self):
        """Yield (file_path, folder_diagnosis, parse_func) tuples, walking subfolders if needed."""
        parse_func = self._get_parser()
        if self.subfolder_label_map:
            for subfolder, label in self.subfolder_label_map.items():
                sub_path = os.path.join(self.path, subfolder)
                if not os.path.isdir(sub_path):
                    continue
                for root, _, files in os.walk(sub_path):
                    for fname in files:
                        if fname.endswith(".cha"):
                            yield os.path.join(root, fname), label, parse_func
        else:
            for root, _, files in os.walk(self.path):
                for fname in files:
                    if fname.endswith(".cha"):
                        label = self._infer_label_from_filename(fname)
                        yield os.path.join(root, fname), label, parse_func

    def __iter__(self) -> Iterator[RawDataPoint]:
        for file_path, folder_diagnosis, parse_func in self._collect_cha_files():
            for datapoint in self.parse_cha_file(file_path, parse_func):
                if self.override_diagnosis:
                    datapoint["Diagnosis"] = self.override_diagnosis
                elif folder_diagnosis and datapoint["Diagnosis"] == "Unknown":
                    datapoint["Diagnosis"] = folder_diagnosis
                yield datapoint

    def parse_cha_file(self, file_path: str, parse_line_func: Callable[[dict, str], None]) -> Iterator[RawDataPoint]:
        """
        Parses a .cha file and yields a single raw data point after processing the entire file.
        """
        info = {
            "age": "Unknown",
            "gender": "Unknown",
            "PID": "Unknown",
            "Languages": "Unknown",
            "Participants": [],
            "File_ID": "Unknown",
            "Media": "Unknown",
            "Education": "Unknown",
            "Modality": "Unknown",
            "Task": [""],
            "Dataset": "Unknown",
            "Diagnosis": "Unknown",
            "MMSE": "Unknown",
            "Continents": "Unknown",
            "Countries": "Unknown",
            "Duration": "Unknown",
            "Location": "Unknown",
            "Date": "Unknown",
            "Transcriber": "Unknown",
            "Moca": "Unknown",
            "Setting": "Unknown",
            "Comment": "Unknown",
            "text_participant": [],
            "text_interviewer":[],
            "text_interviewer_participant": [],
        }


        with open(file_path, 'r') as file:
            for line in file:
                parse_line_func(info, line,file_path)

        # Finalize the text field by joining collected transcript lines
        info["text_participant"] = " ".join(info["text_participant"])
        info["text_interviewer"] = " ".join(info["text_interviewer"])
        info["text_interviewer_participant"] = " ".join(info["text_interviewer_participant"])

        yield info
    
    def _parse_line_general(self, info: dict, line: str, file_path: str):
        """General line parser for English, Chinese, Korean, German, Greek."""
        if line.startswith("@PID:"):
            info["PID"] = line.split(":", 1)[1].strip()
        elif line.startswith("@Date:"):
            info["Date"] = line.split(":", 1)[1].strip()
        elif line.startswith("@Languages:"):
            info["Languages"] = line.split(":", 1)[1].strip()
        elif line.startswith("@Participants:"):
            info["Participants"] = line.split(":", 1)[1].strip()
        elif line.startswith("@Situation:"):
            info["Task"].append("Situation: " + line.split(":", 1)[1].strip())
        elif line.startswith("@Activities:"):
            info["Task"].append("Activities: " + line.split(":", 1)[1].strip())
        elif line.startswith("@Bg:"):
            info["Task"].append(line.split(":", 1)[1].strip())
        elif line.startswith("@G:"):
            info["Task"].append(line.split(":", 1)[1].strip())
        elif line.startswith("@Transcriber:"):
            info["Transcriber"] = line.split(":", 1)[1].strip()
        elif line.startswith("@Location:"):
            info["Location"] = line.split(":", 1)[1].strip()
        elif line.startswith("@Time Duration:"):
            info["Duration"] = line.split(":", 1)[1].strip()
        elif line.startswith("@comment:") or line.startswith("@Comment:"):
            info["Comment"] = line.split(":", 1)[1].strip()
        elif line.startswith("@ID:") and "Participant" in line:
            parts = line.split("|")
            if len(parts) > 5:
                info["Languages"] = parts[0].split()[-1].strip()
                info["Dataset"] = parts[1].strip()
                diag = parts[5].strip()
                if diag:
                    info["Diagnosis"] = diag
                age_info = parts[3].split(';')[0].strip()
                if age_info.isdigit():
                    info["age"] = int(age_info)
                if len(parts) > 4:
                    info["gender"] = parts[4].strip()
                if len(parts) > 9 and parts[9].strip().isdigit():
                    info["MMSE"] = int(parts[9])
                elif len(parts) > 8 and parts[8].strip().isdigit():
                    info["MMSE"] = int(parts[8])
        elif line.startswith("@Media:"):
            media_parts = line.split(":", 1)[1].strip().split(",")
            if len(media_parts) > 1:
                info["File_ID"] = media_parts[0].strip()
                info["Media"] = media_parts[1].strip()
            elif media_parts:
                info["File_ID"] = media_parts[0].strip()
        elif line.startswith("*PAR:") or line.startswith("*PAR0:") or line.startswith("*PAR1:"):
            participant_text = line.split(":", 1)[1].strip()
            info["text_participant"].append(participant_text)
            info["text_interviewer_participant"].append("PAR: " + participant_text)
        elif line.startswith("*INV:"):
            interviewer_text = line.split(":", 1)[1].strip()
            info["text_interviewer"].append(interviewer_text)
            info["text_interviewer_participant"].append("INT: " + interviewer_text)
            

    def _parse_line_spanish(self, info: dict, line: str,file_path: str):
        """
        Language-specific line parser for Spanish.
        """
        info["File_ID"] = os.path.splitext(os.path.basename(file_path))[0]# PerLA
        if line.startswith("@PID:"):
            info["PID"] = line.split(":")[1].strip()
        elif line.startswith("@Transcriber:"):
            info["Transcriber"] = line.split(":")[1].strip()
        elif line.startswith("@Date:"):
            info["Date"] = line.split(":")[1].strip()
        elif line.startswith("@Location:"): 
            info["Location"] = line.split(":")[1].strip()
        elif line.startswith("@Time Duration:"):
            info["Duration"] = line.split(":", 1)[1].strip()
        elif line.startswith("@Languages:"):
            info["Languages"] = line.split(":")[1].strip()
            #info["Languages"]= 'spanish' # Ivanova dataset
        elif line.startswith("@Participants:"): # e.g., PAR Participant, INV Investigator
            info["Participants"] = line.split(":")[1].strip()
        elif line.startswith("@G:"):
            info["Task"].append(line.split(":")[1].strip())
        elif line.startswith("@Situation:"):
            info["Task"].append("Situation: "+line.split(":")[1].strip())
        elif line.startswith("@Bg:"):
            info["Task"].append(line.split(":")[1].strip())
        elif line.startswith("@G:"):
            info["Task"].append(line.split(":")[1].strip())
        elif line.startswith("@comment:"):
            info["Comment"] = line.split(":")[1].strip()
        elif line.startswith("@Media:"):
            media_parts = line.split(":")[1].strip().split(",")
            if len(media_parts) > 1:
                #info["File_ID"] = media_parts[0].strip()# Ivanova
                info["Media"] = media_parts[1].strip()
        elif line.startswith("@ID:") and "Target_Adult" in line:
            parts = line.split("|")
            info["Languages"] = parts[0].split()[-1].strip()
            info["Dataset"] = parts[1].strip()
            info["Diagnosis"] = parts[5].strip()
            age_info = parts[3].split(';')[0].strip()
            if age_info.isdigit():
                info["age"] = int(age_info)
            info["gender"] = parts[4].strip()
            if parts[9].isdigit():
                info["MMSE"] = int(parts[9])
            elif parts[8].isdigit():
                info["MMSE"] = int(parts[8])
        elif line.startswith("@Media:"):
            media_parts = line.split(":")[1].strip().split(",")
            if len(media_parts) > 1:
                info["File_ID"] = media_parts[0].strip()
                info["Media"] = media_parts[1].strip()
        
        elif line.startswith("*PAR:"): # Ivanova
            participant_text = line.replace("*PAR:", "").strip()
            info["text_participant"].append(participant_text)
            info["text_interviewer_participant"].append("PAR: "+participant_text)
        
        elif line.startswith("*"): # PerLA
                 interviewer_text = line
                 info["text_interviewer"].append(interviewer_text)
                 info["text_interviewer_participant"].append(interviewer_text)

        



    def normalize_datapoint(self, raw_datapoint: RawDataPoint) -> NormalizedDataPoint:
        """
        Normalize a raw data point into a standardized format.
        """
        return NormalizedDataPoint(
            PID=raw_datapoint["PID"],
            Languages=raw_datapoint["Languages"],
            MMSE=raw_datapoint["MMSE"],
            Diagnosis=raw_datapoint["Diagnosis"],
            Participants=raw_datapoint["Participants"],
            Dataset=raw_datapoint["Dataset"],
            Modality=raw_datapoint["Media"],
            Task=raw_datapoint["Task"],
            File_ID=raw_datapoint["File_ID"],
            Media=raw_datapoint["Media"],
            Age=raw_datapoint["age"],
            Gender=raw_datapoint["gender"],
            Education=raw_datapoint["Education"],
            Source="CHA Dataset",
            Continents=raw_datapoint["Continents"],
            Countries=raw_datapoint["Countries"],
            Duration=raw_datapoint["Duration"],
            Location=raw_datapoint["Location"],
            Date=raw_datapoint["Date"],
            Transcriber=raw_datapoint["Transcriber"],
            Moca=raw_datapoint["Moca"],
            Setting=raw_datapoint["Setting"],
            Comment=raw_datapoint["Comment"],
            Text_interviewer_participant = raw_datapoint["text_interviewer_participant"],
            Text_participant = raw_datapoint["text_participant"],
            Text_interviewer=raw_datapoint["text_interviewer"]
        )
        





  




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Extract .cha files to JSONL")
    parser.add_argument('--path', required=True, help="Path to folder containing .cha files")
    parser.add_argument('--language', required=True, help="Language: english, chinese, spanish, korean, german, greek")
    parser.add_argument('--output', required=True, help="Output JSONL file path")
    parser.add_argument('--override-diagnosis', default=None, help="Force all samples to this diagnosis label")
    args = parser.parse_args()

    path = os.path.expanduser(args.path)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    collection = CHACollection(path, language=args.language, override_diagnosis=args.override_diagnosis)

    count = 0
    with open(args.output, "w", encoding="utf-8") as outfile:
        for normalized_datapoint in collection.get_normalized_data():
            normalized_dict = asdict(normalized_datapoint)
            json.dump(normalized_dict, outfile, ensure_ascii=False)
            outfile.write("\n")
            count += 1
    print(f"Extracted {count} samples to {args.output}")
            

    

    





