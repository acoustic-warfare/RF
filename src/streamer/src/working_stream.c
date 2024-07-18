#include "gst/gstelementfactory.h"
#include "gst/gstpipeline.h"
#include <gst/gst.h>


static void cb_need_data(GstElement *appsrc, guint unused_size, gpointer user_data) {
  
}


int main(int argc, char *argv[]) {
  GstElement *pipeline, *src_bin, *sink_bin;
  GstElement *src, *sink;
  GstBus *bus;
  GstMessage *msg;
  GstStateChangeReturn ret;

  /* Initialize GStreamer */
  gst_init(&argc, &argv);

  /* Initialize Buses */
  src_bin = gst_bin_new("src-bin");
  sink_bin = gst_bin_new("sink-bin");
  pipeline = gst_pipeline_new("pipeline");
  
  /* Initialize elements */
  src = gst_element_factory_make("videotestsrc", "src");
  sink = gst_element_factory_make("autovideosink", "sink");

  /* Add elements to bins */
  gst_bin_add_many(GST_BIN(src_bin), src, NULL);
  gst_bin_add_many(GST_BIN(sink_bin), sink, NULL);
  gst_bin_add_many(GST_BIN(pipeline), src_bin, sink_bin, NULL);

  /* Link bins */
  gst_element_link(src, sink);

  /* Start playing */
  ret = gst_element_set_state(pipeline, GST_STATE_PLAYING);
  if (ret == GST_STATE_CHANGE_FAILURE) {
    g_printerr("Unable to set the pipeline to the playing state.\n");
    gst_object_unref(pipeline);
    return -1;
  }

  /* Wait until error or EOS */
  bus = gst_element_get_bus(pipeline);
  msg = gst_bus_timed_pop_filtered(bus, GST_CLOCK_TIME_NONE,
                                   GST_MESSAGE_ERROR | GST_MESSAGE_EOS);

  /* See next tutorial for proper error message handling/parsing */
  if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ERROR) {
    g_printerr("An error occurred! Re-run with the GST_DEBUG=*:WARN "
               "environment variable set for more details.\n");
  }

  /* Free resources */
  gst_message_unref(msg);
  gst_object_unref(bus);
  gst_element_set_state(pipeline, GST_STATE_NULL);
  gst_object_unref(pipeline);
}

/*
int tutorial_main(int argc, char *argv[]) {
  GstElement *pipeline;

  // Build the pipeline 
  pipeline = gst_parse_launch(
      "ximagesrc xid=0x1400006 use-damage=0 ! videoconvert ! videoscale ! "
      "video/x-raw,width=1920,height=1080 ! x264enc tune=zerolatency "
      "bitrate=500 speed-preset=superfast ! queue ! flvmux ! rtmp2sink "
      "location=rtmp://ome.waraps.org/app/hej",
      NULL);
  return 0;
}
*/
