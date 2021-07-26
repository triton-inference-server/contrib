package com.nvidia.triton.contrib.pojo;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import com.nvidia.triton.contrib.Util;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * @author xiafei.qiuxf
 * @date 2021/7/26
 */
class IOTensorTest {

    @Test
    void testJson() throws Exception {
        final IOTensor t = new IOTensor();
        t.setName("test_tensor");
        t.setDatatype(DataType.FP32);
        t.setShape(new long[] {1, 2, 3});
        t.setData(new Object[] {9, 8, 7});
        final Parameters param = new Parameters();
        param.put("a", "b");
        t.setParameters(param);

        final String s = Util.toJson(t);

        Map map = Util.fromJson(s, Map.class);
        assertEquals("test_tensor", map.get("name"));
        assertEquals("FP32", map.get("datatype"));
        assertEquals(new HashMap<String, String>() {{
            this.put("a", "b");
        }}, map.get("parameters"));
        assertEquals(Arrays.asList(1, 2, 3), map.get("shape"));
        assertEquals(Arrays.asList(9, 8, 7), map.get("data"));
    }
}