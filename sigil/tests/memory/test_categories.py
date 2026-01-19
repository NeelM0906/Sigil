"""Tests for Layer 3: CategoryLayer and MemoryConsolidator.

Tests cover:
- Category CRUD operations
- Markdown serialization
- Item tracking
- Consolidation quality
- Incremental updates
"""

import pytest
import tempfile
import shutil
from pathlib import Path

from sigil.memory.layers.categories import CategoryLayer
from sigil.memory.consolidation import (
    MemoryConsolidator,
    ConsolidationTrigger,
    CATEGORY_TEMPLATES,
)
from sigil.config.schemas.memory import MemoryCategory, MemoryItem


@pytest.fixture
def temp_storage_dir():
    """Create a temporary storage directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def category_layer(temp_storage_dir):
    """Create a CategoryLayer instance with temporary storage."""
    return CategoryLayer(storage_dir=temp_storage_dir)


@pytest.fixture
def consolidator():
    """Create a MemoryConsolidator instance."""
    return MemoryConsolidator()


class TestCategoryLayerCRUD:
    """Tests for basic category CRUD operations."""

    def test_create_category(self, category_layer):
        """Test creating a new category."""
        category = category_layer.create(
            name="lead_preferences",
            description="Customer preferences and requirements",
        )

        assert category is not None
        assert category.category_id is not None
        assert category.name == "lead_preferences"
        assert category.description == "Customer preferences and requirements"

    def test_create_with_content(self, category_layer):
        """Test creating a category with initial content."""
        content = "# Lead Preferences\n\n## Communication\n- Email preferred"
        category = category_layer.create(
            name="lead_preferences",
            description="Preferences",
            markdown_content=content,
        )

        assert category.markdown_content == content

    def test_create_duplicate_raises_error(self, category_layer):
        """Test that creating a duplicate category raises ValueError."""
        category_layer.create(name="test_category", description="Test")

        with pytest.raises(ValueError) as exc_info:
            category_layer.create(name="test_category", description="Duplicate")
        assert "already exists" in str(exc_info.value)

    def test_get_category(self, category_layer):
        """Test retrieving a category by ID."""
        created = category_layer.create(
            name="test",
            description="Test category",
        )

        retrieved = category_layer.get(created.category_id)

        assert retrieved is not None
        assert retrieved.category_id == created.category_id

    def test_get_by_name(self, category_layer):
        """Test retrieving a category by name."""
        category_layer.create(name="test_category", description="Test")

        retrieved = category_layer.get_by_name("test_category")

        assert retrieved is not None
        assert retrieved.name == "test_category"

    def test_get_by_name_normalized(self, category_layer):
        """Test that name lookup is normalized."""
        category_layer.create(name="test_category", description="Test")

        # Should find with different formats
        assert category_layer.get_by_name("Test Category") is not None
        assert category_layer.get_by_name("test-category") is not None
        assert category_layer.get_by_name("TEST_CATEGORY") is not None

    def test_delete_category(self, category_layer):
        """Test deleting a category."""
        category_layer.create(name="to_delete", description="Test")

        assert category_layer.exists("to_delete")
        deleted = category_layer.delete("to_delete")
        assert deleted is True
        assert not category_layer.exists("to_delete")

    def test_delete_nonexistent(self, category_layer):
        """Test deleting a non-existent category returns False."""
        result = category_layer.delete("nonexistent")
        assert result is False


class TestCategoryContent:
    """Tests for category content management."""

    def test_update_content(self, category_layer):
        """Test updating category content."""
        category_layer.create(
            name="test",
            description="Test",
            markdown_content="Initial content",
        )

        updated = category_layer.update_content(
            name="test",
            markdown_content="Updated content",
        )

        assert updated is not None
        assert updated.markdown_content == "Updated content"

    def test_update_content_append(self, category_layer):
        """Test appending to category content."""
        category_layer.create(
            name="test",
            description="Test",
            markdown_content="Line 1",
        )

        updated = category_layer.update_content(
            name="test",
            markdown_content="Line 2",
            append=True,
        )

        assert "Line 1" in updated.markdown_content
        assert "Line 2" in updated.markdown_content

    def test_get_content(self, category_layer):
        """Test getting just the markdown content."""
        category_layer.create(
            name="test",
            description="Test",
            markdown_content="# Title\n\nContent here",
        )

        content = category_layer.get_content("test")

        assert content is not None
        assert "# Title" in content


class TestCategoryItems:
    """Tests for category item tracking."""

    def test_add_items(self, category_layer):
        """Test adding items to a category."""
        category_layer.create(name="test", description="Test")

        updated = category_layer.add_items(
            name="test",
            item_ids=["item-1", "item-2"],
        )

        assert len(updated.item_ids) == 2
        assert "item-1" in updated.item_ids
        assert "item-2" in updated.item_ids

    def test_add_items_no_duplicates(self, category_layer):
        """Test that duplicate items are not added."""
        category_layer.create(name="test", description="Test")
        category_layer.add_items("test", ["item-1"])

        updated = category_layer.add_items("test", ["item-1", "item-2"])

        assert len(updated.item_ids) == 2

    def test_remove_items(self, category_layer):
        """Test removing items from a category."""
        category_layer.create(name="test", description="Test")
        category_layer.add_items("test", ["item-1", "item-2", "item-3"])

        updated = category_layer.remove_items("test", ["item-2"])

        assert len(updated.item_ids) == 2
        assert "item-2" not in updated.item_ids

    def test_get_item_count(self, category_layer):
        """Test getting the item count for a category."""
        category_layer.create(name="test", description="Test")
        category_layer.add_items("test", ["item-1", "item-2"])

        count = category_layer.get_item_count("test")

        assert count == 2

    def test_get_categories_with_item(self, category_layer):
        """Test finding categories containing a specific item."""
        category_layer.create(name="cat_a", description="A")
        category_layer.create(name="cat_b", description="B")
        category_layer.add_items("cat_a", ["shared-item"])
        category_layer.add_items("cat_b", ["shared-item"])

        categories = category_layer.get_categories_with_item("shared-item")

        assert len(categories) == 2


class TestCategoryListing:
    """Tests for listing and searching categories."""

    def test_list_all(self, category_layer):
        """Test listing all categories."""
        category_layer.create(name="cat_a", description="A")
        category_layer.create(name="cat_b", description="B")

        categories = category_layer.list_all()

        assert len(categories) == 2

    def test_count(self, category_layer):
        """Test counting categories."""
        category_layer.create(name="cat_a", description="A")
        category_layer.create(name="cat_b", description="B")

        assert category_layer.count() == 2

    def test_search_by_content(self, category_layer):
        """Test searching categories by content."""
        category_layer.create(
            name="pricing",
            description="Pricing info",
            markdown_content="Our pricing starts at $500/month",
        )
        category_layer.create(
            name="features",
            description="Features",
            markdown_content="Key features include API access",
        )

        results = category_layer.search_by_content("pricing")

        assert len(results) == 1
        assert results[0].name == "pricing"


class TestMarkdownPersistence:
    """Tests for markdown file persistence."""

    def test_persistence_across_instances(self, temp_storage_dir):
        """Test that categories persist across layer instances."""
        layer1 = CategoryLayer(storage_dir=temp_storage_dir)
        layer1.create(
            name="test",
            description="Test category",
            markdown_content="# Test\n\nContent here",
        )
        layer1.add_items("test", ["item-1", "item-2"])

        layer2 = CategoryLayer(storage_dir=temp_storage_dir)
        retrieved = layer2.get_by_name("test")

        assert retrieved is not None
        assert retrieved.name == "test"
        assert "# Test" in retrieved.markdown_content
        assert len(retrieved.item_ids) == 2

    def test_markdown_file_format(self, temp_storage_dir):
        """Test that markdown files have correct format."""
        layer = CategoryLayer(storage_dir=temp_storage_dir)
        layer.create(
            name="test_category",
            description="Test",
            markdown_content="# Content",
        )

        # Read the file directly
        file_path = Path(temp_storage_dir) / "test_category.md"
        assert file_path.exists()

        content = file_path.read_text()
        assert content.startswith("---")  # YAML frontmatter
        assert "category_id:" in content
        assert "name:" in content
        assert "description:" in content


class TestMemoryConsolidator:
    """Tests for the MemoryConsolidator class."""

    def test_get_template(self, consolidator):
        """Test getting category templates."""
        template = consolidator.get_template("lead_preferences")

        assert template is not None
        assert "description" in template
        assert "template" in template

    def test_get_template_normalized_name(self, consolidator):
        """Test that template lookup normalizes names."""
        assert consolidator.get_template("Lead Preferences") is not None
        assert consolidator.get_template("lead-preferences") is not None

    def test_get_nonexistent_template(self, consolidator):
        """Test that nonexistent template returns None."""
        result = consolidator.get_template("nonexistent_category")
        assert result is None

    def test_should_consolidate(self, consolidator):
        """Test consolidation threshold check."""
        # Create a mock category
        category = MemoryCategory(
            name="test",
            description="Test",
        )

        # Below threshold
        assert not consolidator.should_consolidate(category, 50)

        # At/above threshold (default is 100)
        assert consolidator.should_consolidate(category, 100)
        assert consolidator.should_consolidate(category, 150)

    @pytest.mark.asyncio
    async def test_consolidate_empty_items(self, consolidator):
        """Test consolidation with no items."""
        result = await consolidator.consolidate(
            category_name="test",
            items=[],
        )

        assert result is not None
        assert result.items_consolidated == 0

    @pytest.mark.asyncio
    async def test_consolidate_with_items(self, consolidator):
        """Test consolidation with items (uses fallback)."""
        items = [
            MemoryItem(
                content="Customer prefers email",
                source_resource_id="res-1",
                category="lead_preferences",
            ),
            MemoryItem(
                content="Budget is $500/month",
                source_resource_id="res-2",
                category="budget_information",
            ),
        ]

        result = await consolidator.consolidate(
            category_name="lead_preferences",
            items=items,
        )

        assert result is not None
        assert result.items_consolidated == 2
        assert result.category_name == "lead_preferences"
        assert len(result.markdown_content) > 0

    @pytest.mark.asyncio
    async def test_incremental_consolidation(self, consolidator):
        """Test incremental consolidation."""
        existing_content = "# Existing\n\nPrevious content here"
        new_items = [
            MemoryItem(
                content="New fact",
                source_resource_id="res-1",
            ),
        ]

        result = await consolidator.consolidate_incremental(
            category_name="test",
            new_items=new_items,
            existing_content=existing_content,
        )

        assert result.is_incremental is True

    @pytest.mark.asyncio
    async def test_rebuild_category(self, consolidator):
        """Test full category rebuild."""
        items = [
            MemoryItem(
                content="Fact 1",
                source_resource_id="res-1",
            ),
            MemoryItem(
                content="Fact 2",
                source_resource_id="res-2",
            ),
        ]

        result = await consolidator.rebuild_category(
            category_name="test",
            all_items=items,
        )

        assert result.is_incremental is False
        assert result.trigger == ConsolidationTrigger.MANUAL


class TestCategoryTemplates:
    """Tests for predefined category templates."""

    def test_templates_exist(self):
        """Test that expected templates are defined."""
        expected = [
            "lead_preferences",
            "objection_patterns",
            "product_knowledge",
            "conversation_insights",
            "successful_approaches",
        ]

        for name in expected:
            assert name in CATEGORY_TEMPLATES
            assert "description" in CATEGORY_TEMPLATES[name]
            assert "template" in CATEGORY_TEMPLATES[name]

    def test_templates_have_markdown(self):
        """Test that templates contain valid markdown."""
        for name, template in CATEGORY_TEMPLATES.items():
            content = template["template"]
            assert "#" in content  # Has headers


class TestCreateOrUpdate:
    """Tests for create_or_update functionality."""

    def test_create_new(self, category_layer):
        """Test creating a new category via create_or_update."""
        result = category_layer.create_or_update(
            name="new_category",
            description="New",
            markdown_content="# New",
        )

        assert result.name == "new_category"
        assert "# New" in result.markdown_content

    def test_update_existing(self, category_layer):
        """Test updating existing category via create_or_update."""
        category_layer.create(
            name="existing",
            description="Original",
            markdown_content="# Original",
        )

        result = category_layer.create_or_update(
            name="existing",
            description="Updated",
            markdown_content="# Updated",
        )

        assert result.name == "existing"
        assert "# Updated" in result.markdown_content
