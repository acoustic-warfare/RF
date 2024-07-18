src_main = ['main.c']
src_test = ['working_stream.c']
src_copy = ['test.c']

exe = executable(
  'stream',
  sources: src_main,
  dependencies: [dep_gst],
  include_directories: inc,
)

exe = executable(
  'try_thing',
  sources: src_test,
  dependencies: [dep_gst],
  include_directories: inc,
)

exe = executable(
  'copy',
  sources: src_copy,
  dependencies: [dep_gst],
  include_directories: inc,
)
