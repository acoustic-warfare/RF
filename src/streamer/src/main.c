#include "gst/gstbin.h"
#include "gst/gstelement.h"
#include <glib.h>
#include <gst/gst.h>

inline static void handling_errors(GstMessage *msg) {
  if (msg != NULL) {
    GError *err;
    gchar *debug_info;

    switch (GST_MESSAGE_TYPE(msg)) {
    case GST_MESSAGE_ERROR:
      gst_message_parse_error(msg, &err, &debug_info);
      g_printerr("Error received from element %s: %s\n",
                 GST_OBJECT_NAME(msg->src), err->message);
      g_printerr("Debugging information: %s\n",
                 debug_info ? debug_info : "none");
      g_clear_error(&err);
      g_free(debug_info);
      break;
    case GST_MESSAGE_EOS:
      g_print("End-Of-Stream reached.\n");
      break;
    default:
      /* We should not reach here because we only asked for ERRORs and EOS */
      g_printerr("Unexpected message received.\n");
      break;
    }
    gst_message_unref(msg);
  }
}

int main(int argc, char *argv[]) {
  /*
   * Will hold information about the how all the sources and sincs are
   * connected
   **/
  GstElement *pipeline;

  GstBus *bus;
  GstMessage *msg;
  GstStateChangeReturn ret;

  /* Initialize interal structures of gstreamer */
  gst_init(&argc, &argv);

  /*------------------------------------------------------------------*/
  /* Generate the elements of the pipeline                            */
  /*------------------------------------------------------------------*/

  // Setup Video source
  GstElement *video_src =
      gst_element_factory_make("ximagesrc", "application-screencast");
  g_object_set(video_src, "use-damage", 0,
               NULL); // NOTE: what does this do? Makes it not crash??
  g_object_set(video_src, "xid", 0xa00004, NULL); // Set the screen to record

  GstElement *videoconvert =
      gst_element_factory_make("videoconvert", "video-conversion");

  GstElement *videoscale =
      gst_element_factory_make("videoscale", "resolution-scaler");

  //GstElement *x_raw = gst_element_factory_make("video-raw", );

  // Setup RTMP-sink for sending videostream to RTMP-server
  GstElement *rtmp_sink =
      gst_element_factory_make("autovideosink", "video-sink");
  // g_object_set(rtmp_sink, "location", "rtmp://ome.waraps.org/app/hej", NULL);
  /*------------------------------------------------------------------*/

  /* Initialize the pipeline */
  pipeline = gst_pipeline_new("video-pipeline");

  /* Build the pipline by adding and linking the elements of the pipeline */
  gst_bin_add_many(GST_BIN(pipeline), video_src, videoconvert, videoscale, rtmp_sink,
                   NULL);

  /* The elements need to be linked in the correct order */
  if (gst_element_link_many(video_src, videoconvert, videoscale, rtmp_sink, NULL)) {
    return -1;
  }

  /* Modify the source's properties */

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

  /* Parse message */
  handling_errors(msg);

  /* Free resources */
  gst_object_unref(bus);
  gst_element_set_state(pipeline, GST_STATE_NULL);
  gst_object_unref(pipeline);
  return 0;
}
