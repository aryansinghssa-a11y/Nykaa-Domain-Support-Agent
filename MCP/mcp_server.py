from fastmcp import FastMCP
from order_lookup import check_order_status as task6_check_order_status

mcp = FastMCP("Nykaa Order Lookup MCP Server")


@mcp.tool
def check_order_status(record_id: str) -> dict:
    """
    Look up a Nykaa order using its unique order ID.

    Args:
        record_id: Unique order identifier, e.g. ORD1001.

    Returns:
        Order status, order value and escalation assessment.
    """

    if not isinstance(record_id, str) or not record_id.strip():
        return {
            "found": False,
            "error": "Please provide a valid order ID."
        }

    return task6_check_order_status(record_id.strip())


if __name__ == "__main__":
    mcp.run(
        transport="http",
        host="127.0.0.1",
        port=8001
    )