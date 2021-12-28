*** Settings ***
Documentation     Polaris integration tests
Test Setup    Initialize Server
Test Teardown    Terminate Server
Test Timeout    1200
Library    OperatingSystem
Library    Process
Library    HttpCtrl.Server
Library    ${LIBRARY_PATH}FileComparer.py

*** Variables ***
${LIBRARY_PATH}    libraries/
${NORMFILE}    normalized_frames_integration_test.json
${GRAPHFILE}    graph.json
${GEXFFILE}    graph.gexf
${RESOURCES_PATH}    ${CURDIR}/../resources/
${CHECK_FILES_PATH}    ${RESOURCES_PATH}check_files/
${SATNOGS_NETWORK_URL}    http://127.0.0.1:56000/

*** Test Cases ***
Polaris fetch via import
    ${Result}=    Run Process     polaris fetch --import_file ${RESOURCES_PATH}44420-266-20210409T074243Z-all.csv --cache_dir ./fetch/ LightSail-2 ${NORMFILE}    shell=True
    Should Be Equal As Integers    ${Result.rc}    0
    Should Contain    ${Result.stderr}    Decoding of data finished.
    Should Contain    ${Result.stderr}    Loaded normalizer=Lightsail2
    Should Contain    ${Result.stderr}    Combining space weather and telemetry frames
    Should Contain    ${Result.stderr}    Tagging Completed
    Should Contain    ${Result.stderr}    ${NORMFILE}
    Should Not Contain    ${Result.stderr}    ERROR
    File Should Exist    ${NORMFILE}
    Remove Directory    fetch    recursive=True
    Remove File    ${NORMFILE}

Polaris fetch via download
    Set Environment Variable    SATNOGS_NETWORK_API_URL    ${SATNOGS_NETWORK_URL}
    Start Process    polaris fetch -s 2019-08-11T09:00:00 -e 2019-08-11T12:00:00 --cache_dir ./fetch/ LightSail-2 ${NORMFILE}    shell=True    alias=polaris
    Wait For Request
    ${url}=    Get Request Url
    Should Be Equal    ${url}    /observations/?satellite__norad_cat_id=44420&start=2019-08-11T09%3A00%3A00&end=2019-08-11T12%3A00%3A00&page=1&format=json
    ${response}=    Catenate
    ...    \[
    ...    {
    ...    "start": "2019-08-11T09:37:47Z",
    ...    "end": "2019-08-11T09:50:11Z",
    ...    "demoddata": [
    ...    {
    ...    "payload_demod": "${SATNOGS_NETWORK_URL}media/data_obs/908031/data_908031_2019-08-11T09-42-30"
    ...    }
    ...    ]
    ...    }
    ...    \]
    Reply By   200    ${response}
    Wait For Request    timeout=60
    ${url}=    Get Request Url
    Should Be Equal    ${url}    /observations/?satellite__norad_cat_id=44420&start=2019-08-11T09%3A00%3A00&end=2019-08-11T12%3A00%3A00&page=2&format=json
    Reply By   404
    Wait For Request    timeout=60
    Copy Files    ./resources/demoddata_csv/*.csv   ./fetch/data_2019-08-11_09-00-00/demoddata__08-11-2019T09-00-00__08-11-2019T12-00-00/
    Reply By   302
    ${result}=    Wait For Process    handle=polaris
    Log    ${result}
    Should Be Equal As Integers    ${Result.rc}    0
    Should Contain    ${Result.stderr}    Merge Completed
    Should Contain    ${Result.stderr}    Decoding of data finished.
    Should Contain    ${Result.stderr}    Loaded normalizer=Lightsail2
    Should Contain    ${Result.stderr}    Combining space weather and telemetry frames
    Should Contain    ${Result.stderr}    Tagging Completed
    Should Contain    ${Result.stderr}    ${NORMFILE}
    Should Not Contain    ${Result.stderr}    ERROR
    File Should Exist    ${NORMFILE}
    ${FileCmpResult}=    Compare Files Sizes   ${NORMFILE}    ${CHECK_FILES_PATH}normalized_frames.json
    Should Be True    ${FileCmpResult}
    Remove Directory    fetch    recursive=True

Polaris learn without gridsearch
    ${Result}=    Run Process     polaris learn --force_cpu -g ${GRAPHFILE} ${NORMFILE}    shell=True
    Should Be Equal As Integers    ${Result.rc}    0
    Should Contain    ${Result.stderr}    Plain old gridsearch parameters!
    Should Contain    ${Result.stderr}    No GPU detected! Adding CPU parameters
    Should Contain    ${Result.stderr}    Default thresholds used for dataset na values (col:30% row:60%)
    Should Contain    ${Result.stderr}    Clearing Data. Removing unnecessary columns
    Should Not Contain    ${Result.stderr}    ERROR
    File Should Exist    ${GRAPHFILE}
    ${FileCmpResult}=    Compare Files   ${GRAPHFILE}    ${CHECK_FILES_PATH}graph.json
    Should Be True    ${FileCmpResult}
    Remove Directory    mlruns    recursive=True
    Remove File    ${GRAPHFILE}

Polaris batch
    ${Result}=    Run Process     polaris batch --config_file ${RESOURCES_PATH}polaris_config.json    shell=True
    Should Be Equal As Integers    ${Result.rc}    0
    Should Contain    ${Result.stderr}    Using custom configuration    
    Should Contain    ${Result.stderr}    (col:100% row:100%)
    Should Contain    ${Result.stderr}    Clearing Data. Removing unnecessary columns
    Should Contain    ${Result.stderr}    Dropping constant column(s) : dest_callsign,src_callsign,src_ssid,dest_ssid
    Should Contain    ${Result.stderr}    ${NORMFILE}
    Should Not Contain    ${Result.stderr}    ERROR
    File Should Exist    ${GRAPHFILE}
    ${FileCmpResult}=    Compare Files   ${GRAPHFILE}    ${CHECK_FILES_PATH}graph_batch.json
    Should Be True    ${FileCmpResult}
    Remove Directory    mlruns    recursive=True
    Remove File    ${NORMFILE}

Polaris convert
    ${Result}=    Run Process     polaris convert ${GRAPHFILE} ${GEXFFILE}    shell=True
    Should Be Equal As Integers    ${Result.rc}    0
    Should Not Contain    ${Result.stderr}    ERROR
    File Should Exist    ${GEXFFILE}
    ${FileCmpResult}=    Compare Files   ${GEXFFILE}    ${CHECK_FILES_PATH}convert_graph.gexf
    Should Be True    ${FileCmpResult}
    Remove File    ${GEXFFILE}

Polaris viz
    ${Result}=    Run Process    polaris viz ${GRAPHFILE}    shell=True    timeout=10s
    Should Contain    ${Result.stderr}    Serving ready: http://localhost:8080
    File Should Exist    index.html
    File Should Exist    polaris.js
    File Should Exist    toast.css
    File Should Exist    modal.css
    File Should Exist    style_sheet.css
    File Should Exist    favicon.ico
    Remove File    index.html
    Remove File    polaris.js
    Remove File    toast.css
    Remove File    modal.css
    Remove File    style_sheet.css
    Remove File    favicon.ico
    Remove File    ${GRAPHFILE}

*** Keywords ***
Initialize Server
    Start Server    127.0.0.1    56000

Terminate Server
    Stop Server
