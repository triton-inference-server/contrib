package com.nvidia.triton.contrib;

import java.io.ByteArrayInputStream;
import java.lang.reflect.Method;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import com.google.common.collect.Sets;
import com.google.common.primitives.UnsignedInteger;
import com.google.common.primitives.UnsignedLong;
import com.nvidia.triton.contrib.InferResult.Index;
import com.nvidia.triton.contrib.pojo.DataType;
import com.nvidia.triton.contrib.pojo.IOTensor;
import com.nvidia.triton.contrib.pojo.InferenceResponse;
import com.nvidia.triton.contrib.pojo.Parameters;
import org.apache.http.ProtocolVersion;
import org.apache.http.entity.BasicHttpEntity;
import org.apache.http.message.BasicHttpResponse;
import org.apache.http.message.BasicStatusLine;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * @author xiafei.qiuxf
 * @date 2021/4/20
 */
class InferResultTest {

    private static InferInput createInferInput(String name, DataType dataType, Object data, boolean binary)
        throws Exception {
        long[] shape = {3, 2};
        InferInput input = new InferInput(name, shape, dataType);
        Method setData = InferInput.class.getDeclaredMethod("setData", data.getClass(), boolean.class);
        setData.invoke(input, data, binary);
        return input;
    }

    private static InferResult createResult(DataType dataType, Object data, boolean binary) throws Exception {
        InferInput input = createInferInput("foo", dataType, data, binary);

        InferenceResponse resp = new InferenceResponse();
        resp.setModelName("mo");
        resp.setOutputs(Collections.singletonList(input.getTensor()));
        Parameters param = new Parameters();
        if (binary) {
            param.put("binary_data_size", input.getBinaryData().length);
        }
        resp.setParameters(param);
        Map<String, Index> indexMap = new HashMap<>();
        if (binary) {
            indexMap.put(input.getName(), new Index(0, input.getBinaryData().length));
        }
        return new InferResult(resp, indexMap, input.getBinaryData());
    }

    @ParameterizedTest
    @ValueSource(booleans = {true, false})
    void testGetOutputs(boolean binary) throws Exception {
        InferResult result = createResult(DataType.FP32, new float[] {1F, 1F, 1F, 1F, 1F, 1F}, binary);
        assertEquals(Collections.singletonList("foo"), result.getOutputs());
    }

    @ParameterizedTest
    @ValueSource(booleans = {true, false})
    public void testBinaryBool(boolean binary) throws Exception {
        boolean[] data = {true, false, true, false, true, false};
        InferResult inferResult = createResult(DataType.BOOL, data, binary);
        boolean[] output = inferResult.getOutputAsBool("foo");
        assertArrayEquals(data, output);
        assertNull(inferResult.getOutputAsBool("no"));
    }

    @ParameterizedTest
    @ValueSource(booleans = {true, false})
    public void testBinaryINT8(boolean binary) throws Exception {
        byte[] data = {1, 2, 3, 4, -5, -6};
        InferResult inferResult = createResult(DataType.INT8, data, binary);
        byte[] output = inferResult.getOutputAsByte("foo");
        assertArrayEquals(data, output);
        assertNull(inferResult.getOutputAsByte("no"));
    }

    @ParameterizedTest
    @ValueSource(booleans = {true, false})
    public void testBinaryUINT8(boolean binary) throws Exception {
        byte[] data = {1, 2, 3, 4, -5, -6}; // negative represent large value.
        InferResult inferResult = createResult(DataType.UINT8, data, binary);
        byte[] output = inferResult.getOutputAsByte("foo");
        assertArrayEquals(data, output);
        assertNull(inferResult.getOutputAsByte("no"));
    }

    @ParameterizedTest
    @ValueSource(booleans = {true, false})
    public void testBinaryINT16(boolean binary) throws Exception {
        short[] data = {1, 2, 3, 4, -5, -6};
        InferResult inferResult = createResult(DataType.INT16, data, binary);
        short[] output = inferResult.getOutputAsShort("foo");
        assertArrayEquals(data, output);
        assertNull(inferResult.getOutputAsShort("no"));
    }

    @ParameterizedTest
    @ValueSource(booleans = {true, false})
    public void testBinaryUINT16(boolean binary) throws Exception {
        short[] data = {1, 2, 3, 4, -5, -6};
        InferResult inferResult = createResult(DataType.UINT16, data, binary);
        short[] output = inferResult.getOutputAsShort("foo");
        assertArrayEquals(data, output);
        assertNull(inferResult.getOutputAsShort("no"));
    }

    @ParameterizedTest
    @ValueSource(booleans = {true, false})
    public void testBinaryINT32(boolean binary) throws Exception {
        int[] data = {1, 2, 3, 4, -5, -6};
        InferResult inferResult = createResult(DataType.INT32, data, binary);
        int[] output = inferResult.getOutputAsInt("foo");
        assertArrayEquals(data, output);
        assertNull(inferResult.getOutputAsInt("no"));
    }

    @ParameterizedTest
    @ValueSource(booleans = {true, false})
    public void testBinaryUINT32(boolean binary) throws Exception {
        int[] data = {1, 2, 3, 4,
            UnsignedInteger.valueOf("AFFFFFF", 16).intValue(),
            UnsignedInteger.valueOf("BFFFFFF", 16).intValue()};
        InferResult inferResult = createResult(DataType.UINT32, data, binary);
        int[] output = inferResult.getOutputAsInt("foo");
        assertArrayEquals(data, output);
        assertNull(inferResult.getOutputAsInt("no"));
    }

    @ParameterizedTest
    @ValueSource(booleans = {true, false})
    public void testBinaryINT64(boolean binary) throws Exception {
        long[] data = {1, 2, 3, 4, -5, -6};
        InferResult inferResult = createResult(DataType.INT64, data, binary);
        long[] output = inferResult.getOutputAsLong("foo");
        assertArrayEquals(data, output);
        assertNull(inferResult.getOutputAsLong("no"));
    }

    @ParameterizedTest
    @ValueSource(booleans = {true, false})
    public void testBinaryUINT64(boolean binary) throws Exception {
        long[] data = {1, 2, 3, 4,
            UnsignedLong.valueOf("AFFFFFFFFFFFFFFF", 16).intValue(),
            UnsignedLong.valueOf("BFFFFFFFFFFFFFFF", 16).intValue()};
        InferResult inferResult = createResult(DataType.UINT64, data, binary);
        long[] output = inferResult.getOutputAsLong("foo");
        assertArrayEquals(data, output);
        assertNull(inferResult.getOutputAsLong("no"));
    }

    @ParameterizedTest
    @ValueSource(booleans = {true, false})
    public void testBinaryFP32(boolean binary) throws Exception {
        float[] data = {1.1F, 2.2F, 3.3F, 4.4F, -5.5F, -6.6F};
        InferResult inferResult = createResult(DataType.FP32, data, binary);
        float[] output = inferResult.getOutputAsFloat("foo");
        assertArrayEquals(data, output);
        assertNull(inferResult.getOutputAsFloat("no"));
    }

    @ParameterizedTest
    @ValueSource(booleans = {true, false})
    public void testBinaryFP64(boolean binary) throws Exception {
        double[] data = {1.1F, 2.2F, 3.3F, 4.4F, -5.5F, -6.6F};
        InferResult inferResult = createResult(DataType.FP64, data, binary);
        double[] output = inferResult.getOutputAsDouble("foo");
        assertArrayEquals(data, output);
        assertNull(inferResult.getOutputAsDouble("no"));
    }

    @Test
    void testParseHttpResult_Error() throws Exception {
        BasicHttpResponse resp = new BasicHttpResponse(new BasicStatusLine(new ProtocolVersion("http", 1, 0), 400, ""));

        Throwable ex = assertThrows(IllegalStateException.class,
            () -> new InferResult(resp));
        assertEquals(ex.getMessage(), "Get null entity from HTTP response.");

        BasicHttpEntity entity = new BasicHttpEntity();
        entity.setContent(new ByteArrayInputStream("{".getBytes(StandardCharsets.UTF_8)));
        resp.setEntity(entity);
        ex = assertThrows(InferenceException.class, () -> new InferResult(resp));
        assertEquals(ex.getMessage(), "Malformed error response: {");

        entity.setContent(new ByteArrayInputStream("{\"error\": \"hi\"}".getBytes(StandardCharsets.UTF_8)));
        resp.setEntity(entity);
        ex = assertThrows(InferenceException.class, () -> new InferResult(resp));
        assertEquals(ex.getMessage(), "hi");
    }

    @Test
    void testParseHttpResult_PureJSON() throws Exception {
        IOTensor t1 = new IOTensor();
        t1.setName("t1");
        t1.setShape(new long[] {2, 2});
        t1.setDatatype(DataType.FP32);
        t1.setData(new Object[]{1.1F, 2.2F, 3.3F, 4.4F});

        IOTensor t2 = new IOTensor();
        t2.setName("t2");
        t2.setShape(new long[] {2, 2});
        t2.setDatatype(DataType.INT32);
        t2.setData(new Object[]{1, 2, 3, 4});

        InferenceResponse inferResp = new InferenceResponse();
        inferResp.setModelName("my_model");
        inferResp.setModelVersion("1");
        inferResp.setId("aaa");
        inferResp.setParameters(null);
        inferResp.setOutputs(Arrays.asList(t1, t2));

        BasicHttpResponse httpResp = new BasicHttpResponse(
            new BasicStatusLine(new ProtocolVersion("http", 1, 0), 200, ""));
        BasicHttpEntity entity = new BasicHttpEntity();
        byte[] inferRespBytes = Util.toJson(inferResp).getBytes(StandardCharsets.UTF_8);
        entity.setContent(new ByteArrayInputStream(inferRespBytes));
        httpResp.setEntity(entity);

        InferResult result = new InferResult(httpResp);

        assertEquals(
            Sets.newHashSet("t1", "t2"),
            Sets.newHashSet(result.getOutputs()));
        float[] t1Tensor = result.getOutputAsFloat("t1");
        assertArrayEquals(new float[] {1.1F, 2.2F, 3.3F, 4.4F}, t1Tensor);
        int[] t2Tensor = result.getOutputAsInt("t2");
        assertArrayEquals(new int[] {1, 2, 3, 4}, t2Tensor);
    }

    @Test
    void testParseHttpResult_PureBinary() throws Exception {
        IOTensor t1 = new IOTensor();
        t1.setName("t1");
        t1.setShape(new long[] {2, 2});
        t1.setDatatype(DataType.FP32);
        byte[] bytes1 = BinaryProtocol.toBytes(DataType.FP32, new float[] {1.1F, 2.2F, 3.3F, 4.4F});
        t1.setParameters(new Parameters(new HashMap<String, Object>() {{
            this.put("binary_data_size", bytes1.length);
        }}));

        IOTensor t2 = new IOTensor();
        t2.setName("t2");
        t2.setShape(new long[] {2, 2});
        t2.setDatatype(DataType.INT32);
        byte[] bytes2 = BinaryProtocol.toBytes(DataType.INT32, new int[] {1, 2, 3, 4});
        t2.setParameters(new Parameters(new HashMap<String, Object>() {{
            this.put("binary_data_size", bytes2.length);
        }}));

        InferenceResponse inferResp = new InferenceResponse();
        inferResp.setModelName("my_model");
        inferResp.setModelVersion("1");
        inferResp.setId("aaa");
        inferResp.setOutputs(Arrays.asList(t1, t2));

        BasicHttpResponse httpResp = new BasicHttpResponse(
            new BasicStatusLine(new ProtocolVersion("http", 1, 0), 200, ""));
        BasicHttpEntity entity = new BasicHttpEntity();
        ByteBuffer buf = ByteBuffer.allocate(1024);

        byte[] jsonBytes = Util.toJson(inferResp).getBytes(StandardCharsets.UTF_8);
        buf.put(jsonBytes);
        buf.put(bytes1);
        buf.put(bytes2);
        httpResp.setHeader("Inference-Header-Content-Length", String.valueOf(jsonBytes.length));

        entity.setContent(new ByteArrayInputStream(buf.array(), 0, buf.position()));
        httpResp.setEntity(entity);

        InferResult result = new InferResult(httpResp);

        assertEquals(
            Sets.newHashSet("t1", "t2"),
            Sets.newHashSet(result.getOutputs()));
        float[] t1Tensor = result.getOutputAsFloat("t1");
        assertArrayEquals(new float[] {1.1F, 2.2F, 3.3F, 4.4F}, t1Tensor);
        int[] t2Tensor = result.getOutputAsInt("t2");
        assertArrayEquals(new int[] {1, 2, 3, 4}, t2Tensor);
    }
}