package com.nvidia.triton.contrib.pojo;

import java.util.Arrays;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.nvidia.triton.contrib.Util;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * @author xiafei.qiuxf
 * @date 2021/4/19
 */
public class ParametersTest {

    private static Parameters createParameters() {
        Parameters p1 = new Parameters();
        p1.put("int", 100);
        p1.put("float", 3.22);
        p1.put("str", "abcd");
        p1.put("str_int", "100");
        p1.put("str_float", "3.22");
        return p1;
    }

    @Test
    public void testToJSONMap() throws Exception {
        Parameters p = createParameters();

        String json = Util.toJson(p);
        final ObjectMapper mapper = new ObjectMapper();
        final JsonNode jsonObj = mapper.readTree(json);

        assertTrue(jsonObj.get("int").isInt());
        Object anInt = jsonObj.get("int").doubleValue();
        assertEquals(anInt, 100.0);

        assertTrue(jsonObj.get("float").isFloatingPointNumber());
        double aFloat = jsonObj.get("float").asDouble();
        assertEquals(aFloat, 3.22, 0.00001);

        assertTrue(jsonObj.get("str").isTextual());
        Object str = jsonObj.get("str").asText();
        assertEquals(str, "abcd");
    }

    @Test
    public void testToFromJSON() throws Exception {
        Parameters p1 = createParameters();

        String json = Util.toJson(p1);
        Parameters p2 = Util.fromJson(json, Parameters.class);

        assertEquals(p1.getInt("int"), p2.getInt("int"));
        assertEquals(p1.getFloat("float"), p2.getFloat("float"), 0.00001);
        assertEquals(p1.getString("str"), p2.getString("str"));
    }

    @Test
    public void testGetInt() {
        Parameters p = createParameters();
        assertNull(p.getInt("no"));
        assertEquals(p.getInt("int").intValue(), 100);
        assertEquals(p.getInt("float").intValue(), 3);
        assertEquals(p.getInt("str_int").intValue(), 100);
        assertThrows(IllegalArgumentException.class, () -> {
            createParameters().getInt("str");
        });
    }

    @Test
    public void testGetFloat() {
        Parameters p = createParameters();
        assertNull(p.getFloat("no"));
        assertEquals(p.getFloat("int"), 100.0f, 0.00001);
        assertEquals(p.getFloat("float"), 3.22f, 0.00001);
        assertEquals(p.getFloat("str_float"), 3.22f, 0.00001);
        assertThrows(IllegalArgumentException.class, () -> {
            createParameters().getFloat("str");
        });
    }

    @Test
    public void testGetDouble() {
        Parameters p = createParameters();
        assertNull(p.getDouble("no"));
        assertEquals(p.getDouble("int"), 100.0f, 0.00001);
        assertEquals(p.getDouble("float"), 3.22f, 0.00001);
        assertEquals(p.getDouble("str_float"), 3.22f, 0.00001);
        assertThrows(IllegalArgumentException.class, () -> {
            createParameters().getDouble("str");
        });
    }

    @Test
    public void testGetString() {
        Parameters p = createParameters();
        assertNull(p.getString("no"));
        assertEquals(p.getString("int"), "100");
        assertEquals(p.getString("float"), "3.22");
        assertEquals(p.getString("str"), "abcd");
    }

    @Test
    public void testEmpty() {
        Parameters p = createParameters();
        assertFalse(p.isEmpty());
        for (String key : Arrays.asList("int", "float", "str", "str_int", "str_float")) {
            p.remove(key);
        }
        assertTrue(p.isEmpty());

    }
}