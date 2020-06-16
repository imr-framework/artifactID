from pathlib import Path

import pydicom as pyd
from pydicom.errors import InvalidDicomError
from tqdm import tqdm

arr_deid_tags = ['AccessionNumber', 'AcquisitionDate', 'AcquisitionDatetime', 'AcquisitionTime', 'ContentDate',
                 'ContentTime', 'CountryOfResidence', 'CurrentPatientLocation', 'CurveDate', 'CurveTime', 'Date',
                 'DateTime', 'InstitutionAddress', 'InstitutionName', 'InstitutionalDepartmentName',
                 'IssuerOfPatientID', 'NameOfPhysicianReadingStudy', 'OperatorsName', 'OtherPatientIDs',
                 'OtherPatientNames', 'OverlayDate', 'OverlayTime', 'PatientID', 'PatientsAddress', 'PatientsBirthDate',
                 'PatientsBirthName', 'PatientsBirthTime', 'PatientsInstitutionResidence', 'PatientsMothersBirthName',
                 'PatientName', 'PatientsName', 'PatientsTelephoneNumbers', 'PerformingPhysicianIDSequence',
                 'PerformingPhysicianName', 'PerformingPhysiciansName', 'PersonName', 'PhysicianOfRecord',
                 'PhysicianOfRecordIDSequence', 'PhysicianReadingStudyIDSequence', 'ReferringPhysicianIDSequence',
                 'ReferringPhysiciansAddress', 'ReferringPhysicianName', 'ReferringPhysiciansName',
                 'ReferringPhysiciansTelephoneNumber', 'RegionOfResidence', 'SeriesDate', 'SeriesTime', 'StudyDate',
                 'StudyID', 'StudyTime', 'Time']


def main(path_read: str, path_save_root: str):
    path_read = Path(path_read)
    path_save_root = Path(path_save_root)
    if not path_save_root.exists():
        path_save_root.mkdir(parents=True)

    print('De-identifying DICOMs...')
    arr_dcm = list(path_read.glob('**/*'))
    arr_dcm = list(filter(lambda file: file.is_file(), arr_dcm))  # Exclude folders etc.
    arr_invalid_dcm = []
    for dicom in tqdm(arr_dcm):
        try:
            dcm = pyd.dcmread(str(dicom))
            dcm.remove_private_tags()
            for tag in arr_deid_tags:
                if tag in dcm:
                    data_element = dcm.data_element(tag)
                    data_element.value = 'anonymous'

            # Construct save path
            path_save = path_save_root.joinpath(dicom.relative_to(path_read))
            if not path_save.parent.exists():
                path_save.parent.mkdir(parents=True)
            dcm.save_as(filename=str(path_save))  # Save DICOM
        except InvalidDicomError:
            arr_invalid_dcm.append(dicom)

    print('Invalid DICOMs:')
    print(arr_invalid_dcm)


if __name__ == '__main__':
    path_read = ''
    path_save_root = ''
    main(path_read=path_read, path_save_root=path_save_root)
