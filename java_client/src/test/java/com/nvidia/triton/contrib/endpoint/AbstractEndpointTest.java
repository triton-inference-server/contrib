package com.nvidia.triton.contrib.endpoint;

import java.util.HashSet;
import java.util.List;

import com.google.common.collect.Lists;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

/**
 * @author xiafei.qiuxf
 * @date 2021/4/27
 */
class AbstractEndpointTest {

    private static class FakeEndpoint extends AbstractEndpoint {

        private final int num;
        private final List<String> returnList;

        public FakeEndpoint(List<String> returnList) {
            this.returnList = returnList;
            this.num = new HashSet<>(returnList).size();
        }

        @Override
        String getEndpointImpl() throws Exception {
            return returnList.remove(0);
        }

        @Override
        int getEndpointNum() throws Exception {
            return num;
        }
    }

    @Test
    void nameEndpoint_Fixed() throws Exception {
        FakeEndpoint endpoint = new FakeEndpoint(Lists.newArrayList("a", "a", "a", "a", "a"));
        for (int i = 0; i <5; i++) {
            Assertions.assertEquals("a", endpoint.getEndpoint());
        }
    }

    @Test
    void testEndpoint_RoundRobin() throws Exception {
        FakeEndpoint endpoint = new FakeEndpoint(Lists.newArrayList("a", "b", "a", "b", "a"));
        Assertions.assertEquals("a", endpoint.getEndpoint());
        Assertions.assertEquals("b", endpoint.getEndpoint());
        Assertions.assertEquals("a", endpoint.getEndpoint());
        Assertions.assertEquals("b", endpoint.getEndpoint());
        Assertions.assertEquals("a", endpoint.getEndpoint());
    }

    @Test
    void testEndpoint_WithDuplication() throws Exception {
        FakeEndpoint endpoint = new FakeEndpoint(Lists.newArrayList("a", "a", "b", "a", "b"));
        Assertions.assertEquals("a", endpoint.getEndpoint());
        Assertions.assertEquals("b", endpoint.getEndpoint());
        Assertions.assertEquals("a", endpoint.getEndpoint());
        Assertions.assertEquals("b", endpoint.getEndpoint());
    }
}