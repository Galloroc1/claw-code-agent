from __future__ import annotations

import unittest

from src.core.agent_manager import AgentManager, ManagedAgentGroup, ManagedAgentRecord


class TestManagedAgentRecordDefaults(unittest.TestCase):
    def test_defaults(self) -> None:
        rec = ManagedAgentRecord(agent_id="a1", prompt="do stuff")
        self.assertEqual(rec.agent_id, "a1")
        self.assertEqual(rec.prompt, "do stuff")
        self.assertIsNone(rec.parent_agent_id)
        self.assertIsNone(rec.group_id)
        self.assertIsNone(rec.child_index)
        self.assertIsNone(rec.label)
        self.assertIsNone(rec.resumed_from_session_id)
        self.assertIsNone(rec.session_id)
        self.assertIsNone(rec.session_path)
        self.assertEqual(rec.status, "running")
        self.assertEqual(rec.turns, 0)
        self.assertEqual(rec.tool_calls, 0)
        self.assertIsNone(rec.stop_reason)


class TestManagedAgentGroupDefaults(unittest.TestCase):
    def test_defaults(self) -> None:
        grp = ManagedAgentGroup(group_id="g1")
        self.assertEqual(grp.group_id, "g1")
        self.assertIsNone(grp.label)
        self.assertIsNone(grp.parent_agent_id)
        self.assertEqual(grp.child_agent_ids, ())
        self.assertEqual(grp.strategy, "serial")
        self.assertEqual(grp.status, "running")
        self.assertEqual(grp.completed_children, 0)
        self.assertEqual(grp.failed_children, 0)
        self.assertEqual(grp.batch_count, 0)
        self.assertEqual(grp.max_batch_size, 0)
        self.assertEqual(grp.dependency_skips, 0)


class TestStartAgent(unittest.TestCase):
    def setUp(self) -> None:
        self.mgr = AgentManager()

    def test_increments_counter_and_returns_unique_ids(self) -> None:
        id1 = self.mgr.start_agent(prompt="task1")
        id2 = self.mgr.start_agent(prompt="task2")
        id3 = self.mgr.start_agent(prompt="task3")
        self.assertEqual(id1, "agent_1")
        self.assertEqual(id2, "agent_2")
        self.assertEqual(id3, "agent_3")
        self.assertEqual(len(self.mgr.records), 3)

    def test_record_stored_with_correct_fields(self) -> None:
        aid = self.mgr.start_agent(prompt="hello", label="my-label")
        rec = self.mgr.records[aid]
        self.assertEqual(rec.agent_id, aid)
        self.assertEqual(rec.prompt, "hello")
        self.assertEqual(rec.label, "my-label")
        self.assertEqual(rec.status, "running")

    def test_with_parent_agent_id_tracks_lineage(self) -> None:
        parent = self.mgr.start_agent(prompt="parent")
        child = self.mgr.start_agent(prompt="child", parent_agent_id=parent)
        rec = self.mgr.records[child]
        self.assertEqual(rec.parent_agent_id, parent)

    def test_with_group_id_registers_child(self) -> None:
        gid = self.mgr.start_group(label="grp")
        aid = self.mgr.start_agent(prompt="task", group_id=gid, child_index=0)
        grp = self.mgr.groups[gid]
        self.assertIn(aid, grp.child_agent_ids)
        self.assertEqual(self.mgr.records[aid].group_id, gid)
        self.assertEqual(self.mgr.records[aid].child_index, 0)

    def test_resumed_agents_tracked(self) -> None:
        aid = self.mgr.start_agent(
            prompt="resume", resumed_from_session_id="sess-old-123"
        )
        rec = self.mgr.records[aid]
        self.assertEqual(rec.resumed_from_session_id, "sess-old-123")


class TestStartGroup(unittest.TestCase):
    def setUp(self) -> None:
        self.mgr = AgentManager()

    def test_creates_group_with_strategy(self) -> None:
        gid = self.mgr.start_group(label="batch", strategy="parallel")
        self.assertEqual(gid, "group_1")
        grp = self.mgr.groups[gid]
        self.assertEqual(grp.label, "batch")
        self.assertEqual(grp.strategy, "parallel")
        self.assertEqual(grp.status, "running")

    def test_increments_group_counter(self) -> None:
        g1 = self.mgr.start_group(label="a")
        g2 = self.mgr.start_group(label="b")
        self.assertEqual(g1, "group_1")
        self.assertEqual(g2, "group_2")

    def test_parent_agent_id_stored(self) -> None:
        aid = self.mgr.start_agent(prompt="parent")
        gid = self.mgr.start_group(label="child-group", parent_agent_id=aid)
        self.assertEqual(self.mgr.groups[gid].parent_agent_id, aid)

    def test_default_strategy_is_serial(self) -> None:
        gid = self.mgr.start_group()
        self.assertEqual(self.mgr.groups[gid].strategy, "serial")


class TestRegisterGroupChild(unittest.TestCase):
    def setUp(self) -> None:
        self.mgr = AgentManager()

    def test_adds_agent_to_group(self) -> None:
        gid = self.mgr.start_group(label="g")
        aid = self.mgr.start_agent(prompt="t")
        self.mgr.register_group_child(gid, aid, child_index=0)
        grp = self.mgr.groups[gid]
        self.assertIn(aid, grp.child_agent_ids)
        self.assertEqual(self.mgr.records[aid].group_id, gid)
        self.assertEqual(self.mgr.records[aid].child_index, 0)

    def test_duplicate_does_not_add_twice(self) -> None:
        gid = self.mgr.start_group(label="g")
        aid = self.mgr.start_agent(prompt="t", group_id=gid, child_index=0)
        # Already registered via start_agent; register again
        self.mgr.register_group_child(gid, aid, child_index=0)
        grp = self.mgr.groups[gid]
        self.assertEqual(grp.child_agent_ids.count(aid), 1)

    def test_unknown_group_is_noop(self) -> None:
        aid = self.mgr.start_agent(prompt="t")
        # Should not raise
        self.mgr.register_group_child("nonexistent", aid, child_index=0)

    def test_unknown_agent_does_not_crash(self) -> None:
        gid = self.mgr.start_group(label="g")
        # Agent does not exist; group gets the ID but record update is skipped
        self.mgr.register_group_child(gid, "fake_agent", child_index=0)
        self.assertIn("fake_agent", self.mgr.groups[gid].child_agent_ids)

    def test_updates_child_index_on_record(self) -> None:
        gid = self.mgr.start_group(label="g")
        aid = self.mgr.start_agent(prompt="t")
        self.mgr.register_group_child(gid, aid, child_index=5)
        self.assertEqual(self.mgr.records[aid].child_index, 5)
        self.assertEqual(self.mgr.records[aid].group_id, gid)


class TestFinishAgent(unittest.TestCase):
    def setUp(self) -> None:
        self.mgr = AgentManager()

    def test_marks_completed_with_stats(self) -> None:
        aid = self.mgr.start_agent(prompt="work")
        self.mgr.finish_agent(
            aid,
            session_id="sess-1",
            session_path="/path/sess",
            turns=5,
            tool_calls=12,
            stop_reason="end_turn",
        )
        rec = self.mgr.records[aid]
        self.assertEqual(rec.status, "completed")
        self.assertEqual(rec.session_id, "sess-1")
        self.assertEqual(rec.session_path, "/path/sess")
        self.assertEqual(rec.turns, 5)
        self.assertEqual(rec.tool_calls, 12)
        self.assertEqual(rec.stop_reason, "end_turn")

    def test_preserves_original_fields(self) -> None:
        aid = self.mgr.start_agent(
            prompt="p", parent_agent_id="parent_x", label="lbl"
        )
        self.mgr.finish_agent(
            aid,
            session_id="s",
            session_path="/p",
            turns=1,
            tool_calls=2,
            stop_reason=None,
        )
        rec = self.mgr.records[aid]
        self.assertEqual(rec.prompt, "p")
        self.assertEqual(rec.parent_agent_id, "parent_x")
        self.assertEqual(rec.label, "lbl")

    def test_unknown_agent_is_noop(self) -> None:
        # Should not raise
        self.mgr.finish_agent(
            "unknown_id",
            session_id=None,
            session_path=None,
            turns=0,
            tool_calls=0,
            stop_reason=None,
        )
        self.assertEqual(len(self.mgr.records), 0)


class TestFinishGroup(unittest.TestCase):
    def setUp(self) -> None:
        self.mgr = AgentManager()

    def test_updates_group_status_and_stats(self) -> None:
        gid = self.mgr.start_group(label="g", strategy="parallel")
        self.mgr.finish_group(
            gid,
            status="completed",
            completed_children=3,
            failed_children=1,
            batch_count=2,
            max_batch_size=4,
            dependency_skips=0,
        )
        grp = self.mgr.groups[gid]
        self.assertEqual(grp.status, "completed")
        self.assertEqual(grp.completed_children, 3)
        self.assertEqual(grp.failed_children, 1)
        self.assertEqual(grp.batch_count, 2)
        self.assertEqual(grp.max_batch_size, 4)
        self.assertEqual(grp.dependency_skips, 0)
        # Preserved fields
        self.assertEqual(grp.label, "g")
        self.assertEqual(grp.strategy, "parallel")

    def test_unknown_group_is_noop(self) -> None:
        self.mgr.finish_group(
            "ghost",
            status="completed",
            completed_children=0,
            failed_children=0,
        )
        self.assertEqual(len(self.mgr.groups), 0)

    def test_preserves_child_agent_ids(self) -> None:
        gid = self.mgr.start_group(label="g")
        aid = self.mgr.start_agent(prompt="t", group_id=gid, child_index=0)
        self.mgr.finish_group(
            gid, status="completed", completed_children=1, failed_children=0
        )
        self.assertIn(aid, self.mgr.groups[gid].child_agent_ids)


class TestChildrenOf(unittest.TestCase):
    def setUp(self) -> None:
        self.mgr = AgentManager()

    def test_returns_only_children_of_specified_parent(self) -> None:
        p1 = self.mgr.start_agent(prompt="parent1")
        p2 = self.mgr.start_agent(prompt="parent2")
        c1 = self.mgr.start_agent(prompt="c1", parent_agent_id=p1)
        c2 = self.mgr.start_agent(prompt="c2", parent_agent_id=p1)
        c3 = self.mgr.start_agent(prompt="c3", parent_agent_id=p2)

        children_p1 = self.mgr.children_of(p1)
        children_p2 = self.mgr.children_of(p2)

        self.assertEqual(len(children_p1), 2)
        ids_p1 = {r.agent_id for r in children_p1}
        self.assertEqual(ids_p1, {c1, c2})

        self.assertEqual(len(children_p2), 1)
        self.assertEqual(children_p2[0].agent_id, c3)

    def test_returns_empty_for_no_children(self) -> None:
        aid = self.mgr.start_agent(prompt="solo")
        self.assertEqual(self.mgr.children_of(aid), ())

    def test_returns_empty_for_unknown_parent(self) -> None:
        self.assertEqual(self.mgr.children_of("nonexistent"), ())


class TestGroupChildren(unittest.TestCase):
    def setUp(self) -> None:
        self.mgr = AgentManager()

    def test_returns_sorted_members(self) -> None:
        gid = self.mgr.start_group(label="g")
        a2 = self.mgr.start_agent(prompt="b", group_id=gid, child_index=2)
        a0 = self.mgr.start_agent(prompt="a", group_id=gid, child_index=0)
        a1 = self.mgr.start_agent(prompt="c", group_id=gid, child_index=1)

        children = self.mgr.group_children(gid)
        self.assertEqual(len(children), 3)
        self.assertEqual(children[0].agent_id, a0)
        self.assertEqual(children[1].agent_id, a1)
        self.assertEqual(children[2].agent_id, a2)

    def test_none_child_index_sorted_last(self) -> None:
        gid = self.mgr.start_group(label="g")
        a_none = self.mgr.start_agent(prompt="x", group_id=gid)
        a0 = self.mgr.start_agent(prompt="y", group_id=gid, child_index=0)

        children = self.mgr.group_children(gid)
        self.assertEqual(children[0].agent_id, a0)
        self.assertEqual(children[1].agent_id, a_none)

    def test_empty_for_unknown_group(self) -> None:
        self.assertEqual(self.mgr.group_children("nope"), ())


class TestGroupSummary(unittest.TestCase):
    def setUp(self) -> None:
        self.mgr = AgentManager()

    def test_aggregates_statistics(self) -> None:
        gid = self.mgr.start_group(label="batch", strategy="parallel")
        a1 = self.mgr.start_agent(
            prompt="t1", group_id=gid, child_index=0,
            resumed_from_session_id="old-sess",
        )
        a2 = self.mgr.start_agent(prompt="t2", group_id=gid, child_index=1)
        self.mgr.finish_agent(
            a1, session_id="s1", session_path="/p1",
            turns=3, tool_calls=5, stop_reason="end_turn",
        )
        self.mgr.finish_agent(
            a2, session_id="s2", session_path="/p2",
            turns=2, tool_calls=4, stop_reason="max_turns",
        )
        self.mgr.finish_group(
            gid, status="completed",
            completed_children=2, failed_children=0,
            batch_count=1, max_batch_size=2,
        )

        summary = self.mgr.group_summary(gid)
        assert summary is not None
        self.assertEqual(summary["group_id"], gid)
        self.assertEqual(summary["label"], "batch")
        self.assertEqual(summary["strategy"], "parallel")
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["child_count"], 2)
        self.assertEqual(summary["completed_children"], 2)
        self.assertEqual(summary["failed_children"], 0)
        self.assertEqual(summary["resumed_children"], 1)
        self.assertEqual(summary["batch_count"], 1)
        self.assertEqual(summary["max_batch_size"], 2)
        self.assertEqual(summary["dependency_skips"], 0)
        self.assertEqual(
            summary["stop_reason_counts"],
            {"end_turn": 1, "max_turns": 1},
        )

    def test_running_agents_counted_as_na(self) -> None:
        gid = self.mgr.start_group(label="g")
        self.mgr.start_agent(prompt="t", group_id=gid, child_index=0)
        summary = self.mgr.group_summary(gid)
        assert summary is not None
        self.assertEqual(summary["stop_reason_counts"], {"n/a": 1})

    def test_returns_none_for_unknown_group(self) -> None:
        self.assertIsNone(self.mgr.group_summary("unknown"))


class TestCompletedRecords(unittest.TestCase):
    def setUp(self) -> None:
        self.mgr = AgentManager()

    def test_filters_only_completed(self) -> None:
        a1 = self.mgr.start_agent(prompt="t1")
        a2 = self.mgr.start_agent(prompt="t2")
        a3 = self.mgr.start_agent(prompt="t3")
        self.mgr.finish_agent(
            a1, session_id="s", session_path="/p",
            turns=1, tool_calls=1, stop_reason="done",
        )
        self.mgr.finish_agent(
            a3, session_id="s2", session_path="/p2",
            turns=2, tool_calls=3, stop_reason="done",
        )

        completed = self.mgr.completed_records()
        self.assertEqual(len(completed), 2)
        ids = {r.agent_id for r in completed}
        self.assertEqual(ids, {a1, a3})

    def test_empty_when_none_completed(self) -> None:
        self.mgr.start_agent(prompt="running")
        self.assertEqual(self.mgr.completed_records(), ())


class TestSummaryLines(unittest.TestCase):
    def setUp(self) -> None:
        self.mgr = AgentManager()

    def test_empty_manager(self) -> None:
        lines = self.mgr.summary_lines()
        self.assertIn("- Managed agents: 0", lines)
        self.assertIn("- Completed agents: 0", lines)
        self.assertIn("- Child agents: 0", lines)
        self.assertIn("- Resumed agents: 0", lines)
        self.assertIn("- Agent groups: 0", lines)
        self.assertIn("- Completed groups: 0", lines)

    def test_basic_output_format(self) -> None:
        a1 = self.mgr.start_agent(prompt="task", label="worker-1")
        self.mgr.finish_agent(
            a1, session_id="s1", session_path="/p",
            turns=4, tool_calls=10, stop_reason="end_turn",
        )
        lines = self.mgr.summary_lines()
        self.assertIn("- Managed agents: 1", lines)
        self.assertIn("- Completed agents: 1", lines)
        # Agent detail line
        detail = [l for l in lines if "worker-1" in l]
        self.assertEqual(len(detail), 1)
        self.assertIn("status=completed", detail[0])
        self.assertIn("turns=4", detail[0])
        self.assertIn("tool_calls=10", detail[0])
        self.assertIn("stop=end_turn", detail[0])

    def test_group_info_in_agent_line(self) -> None:
        gid = self.mgr.start_group(label="g")
        self.mgr.start_agent(prompt="t", group_id=gid, child_index=0, label="child-0")
        lines = self.mgr.summary_lines()
        detail = [l for l in lines if "child-0" in l]
        self.assertEqual(len(detail), 1)
        self.assertIn(f"group={gid}", detail[0])
        self.assertIn("child_index=0", detail[0])

    def test_resumed_from_in_agent_line(self) -> None:
        self.mgr.start_agent(
            prompt="t", label="res",
            resumed_from_session_id="old-sess-id",
        )
        lines = self.mgr.summary_lines()
        detail = [l for l in lines if "res" in l]
        self.assertTrue(any("resumed_from=old-sess-id" in l for l in detail))

    def test_agent_without_label_uses_id(self) -> None:
        aid = self.mgr.start_agent(prompt="no label")
        lines = self.mgr.summary_lines()
        detail = [l for l in lines if aid in l]
        self.assertEqual(len(detail), 1)

    def test_truncation_at_8_agents(self) -> None:
        for i in range(10):
            self.mgr.start_agent(prompt=f"task-{i}")
        lines = self.mgr.summary_lines()
        self.assertIn("- Managed agents: 10", lines)
        plus_line = [l for l in lines if "plus" in l and "managed agents" in l]
        self.assertEqual(len(plus_line), 1)
        self.assertIn("2 more managed agents", plus_line[0])

    def test_truncation_at_6_groups(self) -> None:
        for i in range(8):
            self.mgr.start_group(label=f"grp-{i}")
        lines = self.mgr.summary_lines()
        plus_line = [l for l in lines if "plus" in l and "agent groups" in l]
        self.assertEqual(len(plus_line), 1)
        self.assertIn("2 more agent groups", plus_line[0])

    def test_group_summary_line_format(self) -> None:
        gid = self.mgr.start_group(label="my-batch", strategy="parallel")
        a1 = self.mgr.start_agent(prompt="t1", group_id=gid, child_index=0)
        self.mgr.finish_agent(
            a1, session_id="s", session_path="/p",
            turns=1, tool_calls=2, stop_reason="end_turn",
        )
        self.mgr.finish_group(
            gid, status="completed",
            completed_children=1, failed_children=0,
            batch_count=1, max_batch_size=1,
        )
        lines = self.mgr.summary_lines()
        grp_line = [l for l in lines if "my-batch" in l and "group_status" in l]
        self.assertEqual(len(grp_line), 1)
        self.assertIn("group_status=completed", grp_line[0])
        self.assertIn("children=1", grp_line[0])
        self.assertIn("completed=1", grp_line[0])
        self.assertIn("failed=0", grp_line[0])
        self.assertIn("strategy=parallel", grp_line[0])
        self.assertIn("stop_reasons=end_turn:1", grp_line[0])

    def test_child_and_resumed_counts(self) -> None:
        p = self.mgr.start_agent(prompt="parent")
        self.mgr.start_agent(prompt="c1", parent_agent_id=p)
        self.mgr.start_agent(
            prompt="c2", parent_agent_id=p,
            resumed_from_session_id="old",
        )
        lines = self.mgr.summary_lines()
        self.assertIn("- Child agents: 2", lines)
        self.assertIn("- Resumed agents: 1", lines)


class TestMultipleAgentsAndGroupsInteraction(unittest.TestCase):
    """End-to-end scenario with multiple groups and cross-references."""

    def test_full_lifecycle(self) -> None:
        mgr = AgentManager()

        # Parent agent spawns two groups
        parent = mgr.start_agent(prompt="orchestrate", label="orchestrator")
        g1 = mgr.start_group(label="build", parent_agent_id=parent, strategy="serial")
        g2 = mgr.start_group(label="test", parent_agent_id=parent, strategy="parallel")

        # Group 1 children
        b1 = mgr.start_agent(prompt="build-fe", group_id=g1, child_index=0, parent_agent_id=parent)
        b2 = mgr.start_agent(prompt="build-be", group_id=g1, child_index=1, parent_agent_id=parent)

        # Group 2 children, one resumed
        t1 = mgr.start_agent(
            prompt="test-unit", group_id=g2, child_index=0,
            parent_agent_id=parent, resumed_from_session_id="old-session",
        )
        t2 = mgr.start_agent(prompt="test-e2e", group_id=g2, child_index=1, parent_agent_id=parent)

        # Finish agents
        for aid, turns, tc, sr in [
            (b1, 3, 8, "end_turn"),
            (b2, 4, 10, "end_turn"),
            (t1, 2, 5, "end_turn"),
            (t2, 6, 15, "max_turns"),
        ]:
            mgr.finish_agent(aid, session_id=f"s-{aid}", session_path=f"/p/{aid}", turns=turns, tool_calls=tc, stop_reason=sr)

        mgr.finish_group(g1, status="completed", completed_children=2, failed_children=0, batch_count=2, max_batch_size=1)
        mgr.finish_group(g2, status="completed", completed_children=1, failed_children=1, batch_count=1, max_batch_size=2, dependency_skips=1)

        # Verify children_of
        children = mgr.children_of(parent)
        self.assertEqual(len(children), 4)

        # Verify group_children ordering
        g1_children = mgr.group_children(g1)
        self.assertEqual(g1_children[0].agent_id, b1)
        self.assertEqual(g1_children[1].agent_id, b2)

        g2_children = mgr.group_children(g2)
        self.assertEqual(g2_children[0].agent_id, t1)
        self.assertEqual(g2_children[1].agent_id, t2)

        # Verify completed records (parent is still running)
        completed = mgr.completed_records()
        self.assertEqual(len(completed), 4)

        # Verify group summaries
        s1 = mgr.group_summary(g1)
        assert s1 is not None
        self.assertEqual(s1["child_count"], 2)
        self.assertEqual(s1["resumed_children"], 0)
        self.assertEqual(s1["dependency_skips"], 0)

        s2 = mgr.group_summary(g2)
        assert s2 is not None
        self.assertEqual(s2["child_count"], 2)
        self.assertEqual(s2["resumed_children"], 1)
        self.assertEqual(s2["dependency_skips"], 1)
        self.assertEqual(s2["stop_reason_counts"], {"end_turn": 1, "max_turns": 1})

        # Verify summary_lines produces output
        lines = mgr.summary_lines()
        self.assertIn("- Managed agents: 5", lines)
        self.assertIn("- Completed agents: 4", lines)
        self.assertIn("- Child agents: 4", lines)
        self.assertIn("- Resumed agents: 1", lines)
        self.assertIn("- Agent groups: 2", lines)
        self.assertIn("- Completed groups: 2", lines)


class TestFrozenDataclasses(unittest.TestCase):
    def test_record_is_frozen(self) -> None:
        rec = ManagedAgentRecord(agent_id="a", prompt="p")
        with self.assertRaises(AttributeError):
            rec.status = "completed"  # type: ignore[misc]

    def test_group_is_frozen(self) -> None:
        grp = ManagedAgentGroup(group_id="g")
        with self.assertRaises(AttributeError):
            grp.status = "completed"  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
