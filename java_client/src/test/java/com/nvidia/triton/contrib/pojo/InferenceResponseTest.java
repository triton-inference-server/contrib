package com.nvidia.triton.contrib.pojo;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.nvidia.triton.contrib.Util;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;

public class InferenceResponseTest {

    private static InferenceResponse getInferenceResponse() {
        InferenceResponse resp = new InferenceResponse();
        resp.setOutputs(new ArrayList<>());
        for (String name : Arrays.asList("a", "b")) {
            IOTensor tensor = new IOTensor();
            tensor.setName(name);
            resp.getOutputs().add(tensor);
        }
        return resp;
    }

    @Test
    public void testGetOutputByName() {
        InferenceResponse resp = getInferenceResponse();
        IOTensor tensorB = resp.getOutputByName("b");
        assertEquals(tensorB, resp.getOutputs().get(1));
        assertNull(resp.getOutputByName("c"));
    }

    @Test
    public void testJson() throws Exception {
        InferenceResponse resp = getInferenceResponse();
        final String s = Util.toJson(resp);
        Map map = Util.fromJson(s, Map.class);
        final List<Map<String, String>> outputs = (List<Map<String, String>>)map.get("outputs");
        assertEquals(new HashMap<String, String>() {{
            this.put("name", "a");
        }}, outputs.get(0));
        assertEquals(new HashMap<String, String>() {{
            this.put("name", "b");
        }}, outputs.get(1));
    }
}