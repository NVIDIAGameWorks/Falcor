# copy_luts.cmake (called by the rule)
cmake_path(NORMAL_PATH SRC)
cmake_path(NORMAL_PATH DST)

if(WIN32)
  # Call via cmd so wildcards behave right
  # Quiet flags: /NFL (no file list) /NDL (no dir list) /NJH (no job header) /NJS (no job summary) /NP (no progress)
  execute_process(
    COMMAND cmd /c robocopy
            "${SRC}" "${DST}" *.png *.PNG
            /S /XO /XN /XC /R:1 /W:1 /MT:32
            /NFL /NDL /NJH /NJS /NP
    RESULT_VARIABLE rc
  )
  # robocopy returns 0..7 for success
  if(NOT rc EQUAL 0 AND NOT rc LESS_EQUAL 7)
    message(FATAL_ERROR "robocopy failed, rc=${rc}")
  endif()
else()
  execute_process(
    COMMAND rsync -a --prune-empty-dirs --quiet
                   --include="*/" --include="*.png" --include="*.PNG" --exclude="*"
                   "${SRC}/" "${DST}/"
    RESULT_VARIABLE rc
  )
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "rsync failed, rc=${rc}")
  endif()
endif()
