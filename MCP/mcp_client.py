import asyncio
import json
from fastmcp import Client

MCP_SERVER_URL = "http://127.0.0.1:8001/mcp"

async def main():
    # Client connects to the MCP HTTP endpoint, not the bare host:port root.
    async with Client(MCP_SERVER_URL) as client:
        tools = await client.list_tools()
        tool_names = [tool.name for tool in tools]

        print("TOOLS DISCOVERED:", tool_names)
        assert "check_order_status" in tool_names, (
            "The MCP server did not expose check_order_status."
        )

        # Two distinct order IDs are required by the capstone brief.
        for record_id in ("ORD1001", "ORD1002"):
            print("\\n" + "=" * 65)
            print("MCP TOOL CALL:", record_id)

            result = await client.call_tool(
                "check_order_status",
                {"record_id": record_id}
            )

            # Print the standardized MCP result object as JSON when possible.
            if hasattr(result, "model_dump"):
                output = result.model_dump(mode="json")
            elif hasattr(result, "dict"):
                output = result.dict()
            else:
                output = str(result)

            print(json.dumps(output, indent=2, ensure_ascii=False, default=str))

            # FastMCP CallToolResult commonly exposes structured_content.
            structured = getattr(result, "structured_content", None)
            if structured is not None:
                print("STRUCTURED CONTENT:")
                print(json.dumps(structured, indent=2, ensure_ascii=False, default=str))

            # If tool execution returned an error, fail the client test.
            if getattr(result, "is_error", False):
                raise RuntimeError(f"MCP tool call failed for {record_id}")

    print("\\nPASS: MCP client successfully called check_order_status for two IDs.")

if __name__ == "__main__":
    asyncio.run(main())
