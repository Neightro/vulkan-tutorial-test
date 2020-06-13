#[macro_use]
extern crate vulkano;
extern crate vulkano_win;
extern crate winit;
extern crate image;
extern crate tobj;
extern crate cgmath;

use std::sync::{Arc, RwLock};
use std::collections::HashSet;
use std::thread;
use std::time::{Instant, Duration};

use winit::window::{WindowBuilder, Window};
use winit::dpi::LogicalSize;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{EventLoop, ControlFlow};
use vulkano_win::VkSurfaceBuild;

use vulkano::instance::{
    Instance,
    InstanceExtensions,
    ApplicationInfo,
    Version,
    layers_list,
    PhysicalDevice,
};
use vulkano::instance::debug::{DebugCallback, MessageType, MessageSeverity};
use vulkano::device::{Device, DeviceExtensions, Queue, Features};
use vulkano::swapchain::{
    Surface,
    Capabilities,
    ColorSpace,
    SupportedPresentModes,
    PresentMode,
    FullscreenExclusive,
    Swapchain,
    CompositeAlpha,
    acquire_next_image,
    AcquireError,
};
use vulkano::format::{Format, ClearValue};
use vulkano::image::{
    ImageUsage,
    ImmutableImage,
    Dimensions,
    AttachmentImage,
    ImageDimensions,
    ImageLayout,
    MipmapsCount,
    ImageAccess,
    swapchain::SwapchainImage,
};
use vulkano::sync::{
    self,
    SharingMode,
    GpuFuture,
};
use vulkano::pipeline::{
    GraphicsPipeline,
    GraphicsPipelineAbstract,
    viewport::Viewport,
    depth_stencil::DepthStencil
};
use vulkano::framebuffer::{
    RenderPassAbstract,
    Subpass,
    FramebufferAbstract,
    Framebuffer,
};
use vulkano::command_buffer::{
    AutoCommandBuffer,
    AutoCommandBufferBuilder,
    DynamicState,
    CommandBuffer,
};
use vulkano::buffer::{
    immutable::ImmutableBuffer,
    BufferUsage,
    BufferAccess,
    TypedBufferAccess,
    CpuAccessibleBuffer
};
use vulkano::descriptor::descriptor::{
    DescriptorDesc,
    DescriptorDescTy,
    DescriptorBufferDesc,
    DescriptorImageDesc,
    DescriptorImageDescDimensions,
    DescriptorImageDescArray,
    ShaderStages,
};
use vulkano::descriptor::descriptor_set::{
    UnsafeDescriptorSetLayout,
    FixedSizeDescriptorSetsPool,
    FixedSizeDescriptorSet,
    PersistentDescriptorSetBuf,
    PersistentDescriptorSetImg,
    PersistentDescriptorSetSampler,
};
use vulkano::sampler::{
    Sampler,
    Filter,
    MipmapMode,
    SamplerAddressMode,
};

use image::GenericImageView;

use cgmath::{
    Rad,
    Deg,
    Matrix4,
    Vector3,
    Point3
};

const WIDTH:        u32 = 800;
const HEIGHT:       u32 = 600;
const FRAMERATE:    u32 = 60;
const FRAME_PERIOD: Duration = Duration::from_millis((1.0/FRAMERATE as f32 * 1000.0) as u64);

const VALIDATION_LAYERS: &[&str] =  &[
    "VK_LAYER_LUNARG_standard_validation"
];

const TEXTURE_PATH: &str = "src/27_texture.jpg";
const MODEL_PATH: &str = "src/27_model.obj";

/// Required device extensions
fn device_extensions() -> DeviceExtensions {
    DeviceExtensions {
        khr_swapchain: true,
        .. vulkano::device::DeviceExtensions::none()
    }
}

#[cfg(all(debug_assertions))]
const ENABLE_VALIDATION_LAYERS: bool = true;
#[cfg(not(debug_assertions))]
const ENABLE_VALIDATION_LAYERS: bool = false;

struct QueueFamilyIndices {
    graphics_family: i32,
    present_family: i32,
}
impl QueueFamilyIndices {
    fn new() -> Self {
        Self { graphics_family: -1, present_family: -1 }
    }

    fn is_complete(&self) -> bool {
        self.graphics_family >= 0 && self.present_family >= 0
    }
}

#[derive(Copy, Clone, Default)]
struct Vertex {
    pos: [f32; 3],
    color: [f32; 3],
    tex: [f32; 2],
}
impl Vertex {
    fn new(pos: [f32; 3], color: [f32; 3], tex: [f32; 2]) -> Self {
        Self { pos, color, tex }
    }
}
impl_vertex!(Vertex, pos, color, tex);

#[allow(dead_code)]
#[derive(Copy, Clone)]
struct UniformBufferObject {
    model: Matrix4<f32>,
    view: Matrix4<f32>,
    proj: Matrix4<f32>,
}

type DescriptorSetUBO       = PersistentDescriptorSetBuf<Arc<CpuAccessibleBuffer<UniformBufferObject>>>;
type DescriptorSetImage     = PersistentDescriptorSetImg<Arc<ImmutableImage<Format>>>;
type DescriptorSetResources = ((((), DescriptorSetUBO), DescriptorSetImage), PersistentDescriptorSetSampler);
type DescriptorSet          = Arc<FixedSizeDescriptorSet<DescriptorSetResources>>;

struct Display {
    event_loop: EventLoop<()>,
}

impl Display {
    pub fn new() -> Self {
        Self { event_loop: EventLoop::new() }
    }

    pub fn run(self, should_exit: Arc<RwLock<bool>>) {
        self.event_loop.run(move |event, _target, control_flow| {
            let mut exit: Option<bool> = None;

            // Handle events
            match event {
                Event::WindowEvent { event, .. } => {
                    match event {
                        WindowEvent::CloseRequested => {
                            if let Ok(mut guard) = should_exit.try_write() { *guard = true; }
                            exit = Some(true);
                        },
                        _ => {}
                    }
                }
                _ => {}
            }

            // Read the guard if it has not been locked on already
            if let None = exit {
                exit = Some(match should_exit.read() {
                    Ok(guard) => *guard,
                    Err(_) => true // Another thread is either locked on write or is
                });                // in an error state; assume true in those cases
            }

            *control_flow = if let Some(true) = exit { ControlFlow::Exit } else { ControlFlow::Poll };
        });
    }
}

#[allow(unused)]
struct Renderer {
    instance: Arc<Instance>,
    debug_callback: Option<DebugCallback>,

    surface: Arc<Surface<Window>>,

    physical_device_index: usize, // can't store PhysicalDevice directly (lifetime issues)
    device: Arc<Device>,

    graphics_queue: Arc<Queue>,
    present_queue: Arc<Queue>,

    swap_chain: Arc<Swapchain<Window>>,
    swap_chain_images: Vec<Arc<SwapchainImage<Window>>>,

    render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
    graphics_pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,

    swap_chain_framebuffers: Vec<Arc<dyn FramebufferAbstract + Send + Sync>>,

    vertex_buffer: Arc<dyn BufferAccess + Send + Sync>,
    index_buffer: Arc<dyn TypedBufferAccess<Content=[u32]> + Send + Sync>,
    uniform_buffers: Vec<Arc<CpuAccessibleBuffer<UniformBufferObject>>>,

    descriptor_sets: Vec<DescriptorSet>,

    command_buffers: Vec<Arc<AutoCommandBuffer>>,

    depth_format: Format,

    recreate_swap_chain: bool,
    start_time: Instant,
}

impl Renderer {
    pub fn initialize(display: &Display) -> Self {
        let instance = Self::create_instance();
        let debug_callback = Self::setup_debug_callback(&instance);
        let surface = Self::create_surface(&instance, &display.event_loop);

        let physical_device_index = Self::pick_physical_device(&instance, &surface);
        let (device, graphics_queue, present_queue) = Self::create_logical_device(
            &instance, &surface, physical_device_index);

        let (swap_chain, swap_chain_images) = Self::create_swap_chain(&instance, &surface, physical_device_index,
            &device, &graphics_queue, &present_queue);

        let depth_format = Self::find_depth_format();
        let depth_image = Self::create_depth_image(&device, swap_chain.dimensions(), depth_format);

        let render_pass = Self::create_render_pass(&device, swap_chain.format(), depth_format);

        let graphics_pipeline = Self::create_graphics_pipeline(&device, swap_chain.dimensions(), &render_pass);

        let swap_chain_framebuffers = Self::create_framebuffers(&swap_chain_images, &render_pass, &depth_image);

        let start_time = Instant::now();

        let texture_image = Self::create_texture_image(&graphics_queue);
        let image_sampler = Self::create_image_sampler(&device);

        let (vertices, indices) = Self::load_model();

        let vertex_buffer = Self::create_vertex_buffer(&graphics_queue, vertices);
        let index_buffer = Self::create_index_buffer(&graphics_queue, indices);
        let uniform_buffers = Self::create_uniform_buffers(&device, swap_chain_images.len(), start_time, swap_chain.dimensions());

        let descriptor_sets = Self::create_descriptor_sets(&device, &uniform_buffers, &texture_image, &image_sampler);

        let mut app = Self {
            instance,
            debug_callback,
            
            surface,

            physical_device_index,
            device,

            graphics_queue,
            present_queue,

            swap_chain,
            swap_chain_images,

            render_pass,
            graphics_pipeline,

            swap_chain_framebuffers,

            vertex_buffer,
            index_buffer,
            uniform_buffers,

            descriptor_sets,

            command_buffers: vec![],

            depth_format,

            recreate_swap_chain: false,
            start_time
        };

        app.create_command_buffers();
        app
    }

    fn create_instance() -> Arc<Instance> {
        if ENABLE_VALIDATION_LAYERS && !Self::check_validation_layer_support() {
            println!("Validation layers requested, but not available!")
        }

        let supported_extensions = InstanceExtensions::supported_by_core()
            .expect("failed to retrieve supported extensions");
        println!("Supported extensions: {:?}", supported_extensions);

        let app_info = ApplicationInfo {
            application_name: Some("Hello Triangle".into()),
            application_version: Some(Version { major: 1, minor: 0, patch: 0 }),
            engine_name: Some("No Engine".into()),
            engine_version: Some(Version { major: 1, minor: 0, patch: 0 }),
        };

        let required_extensions = Self::get_required_extensions();

        if ENABLE_VALIDATION_LAYERS && Self::check_validation_layer_support() {
            Instance::new(Some(&app_info), &required_extensions, VALIDATION_LAYERS.iter().cloned())
                .expect("failed to create Vulkan instance")
        } else {
            Instance::new(Some(&app_info), &required_extensions, None)
                .expect("failed to create Vulkan instance")
        }
    }

    fn check_validation_layer_support() -> bool {
        let layers: Vec<_> = layers_list().unwrap().map(|l| l.name().to_owned()).collect();
        VALIDATION_LAYERS.iter()
            .all(|layer_name| layers.contains(&layer_name.to_string()))
    }

    fn get_required_extensions() -> InstanceExtensions {
        let mut extensions = vulkano_win::required_extensions();
        if ENABLE_VALIDATION_LAYERS { extensions.ext_debug_utils = true; }
        extensions
    }

    fn setup_debug_callback(instance: &Arc<Instance>) -> Option<DebugCallback> {
        if !ENABLE_VALIDATION_LAYERS  { return None; }

        let msg_types = MessageType::all();
        let severity = MessageSeverity {
            error: true,
            warning: true,
            information: true,
            verbose: false,
        };
        DebugCallback::new(&instance, severity, msg_types, |msg| {
            println!("validation layer: {:?}", msg.description);
        }).ok()
    }

    fn pick_physical_device(instance: &Arc<Instance>, surface: &Arc<Surface<Window>>) -> usize {
        PhysicalDevice::enumerate(&instance)
            .position(|device| Self::is_device_suitable(surface, &device))
            .expect("failed to find a suitable GPU!")
    }

    fn is_device_suitable(surface: &Arc<Surface<Window>>, device: &PhysicalDevice) -> bool {
        let indices = Self::find_queue_families(surface, device);
        let extensions_supported = Self::check_device_extension_support(device);

        let swap_chain_adequate = if extensions_supported {
                let capabilities = surface.capabilities(*device)
                    .expect("failed to get surface capabilities");
                !capabilities.supported_formats.is_empty() &&
                    capabilities.present_modes.iter().next().is_some()
            } else {
                false
            };

        indices.is_complete() && extensions_supported && swap_chain_adequate
    }

    fn check_device_extension_support(device: &PhysicalDevice) -> bool {
        let available_extensions = DeviceExtensions::supported_by_device(*device);
        let device_extensions = device_extensions();
        available_extensions.intersection(&device_extensions) == device_extensions
    }

    fn choose_swap_surface_format(available_formats: &[(Format, ColorSpace)]) -> (Format, ColorSpace) {
        // NOTE: the 'preferred format' mentioned in the tutorial doesn't seem to be
        // queryable in Vulkano (no VK_FORMAT_UNDEFINED enum)
        *available_formats.iter()
            .find(|(format, color_space)|
                *format == Format::B8G8R8A8Unorm && *color_space == ColorSpace::SrgbNonLinear
            )
            .unwrap_or_else(|| &available_formats[0])
    }

    fn choose_swap_present_mode(available_present_modes: SupportedPresentModes) -> PresentMode {
        if available_present_modes.mailbox { PresentMode::Mailbox }
        else if available_present_modes.immediate { PresentMode::Immediate }
        else { PresentMode::Fifo }
    }

    fn choose_swap_extent(capabilities: &Capabilities) -> [u32; 2] {
        if let Some(current_extent) = capabilities.current_extent {
            return current_extent
        } else {
            let mut actual_extent = [WIDTH, HEIGHT];
            actual_extent[0] = capabilities.min_image_extent[0]
                .max(capabilities.max_image_extent[0].min(actual_extent[0]));
            actual_extent[1] = capabilities.min_image_extent[1]
                .max(capabilities.max_image_extent[1].min(actual_extent[1]));
            actual_extent
        }
    }

    fn create_swap_chain(
        instance: &Arc<Instance>,
        surface: &Arc<Surface<Window>>,
        physical_device_index: usize,
        device: &Arc<Device>,
        graphics_queue: &Arc<Queue>,
        present_queue: &Arc<Queue>,
    ) -> (Arc<Swapchain<Window>>, Vec<Arc<SwapchainImage<Window>>>) {
        let physical_device = PhysicalDevice::from_index(&instance, physical_device_index).unwrap();
        let capabilities = surface.capabilities(physical_device)
            .expect("failed to get surface capabilities");

        let surface_format = Self::choose_swap_surface_format(&capabilities.supported_formats);
        let present_mode = Self::choose_swap_present_mode(capabilities.present_modes);
        let extent = Self::choose_swap_extent(&capabilities);

        let mut image_count = capabilities.min_image_count + 1;
        if capabilities.max_image_count.is_some() && image_count > capabilities.max_image_count.unwrap() {
            image_count = capabilities.max_image_count.unwrap();
        }

        let image_usage = ImageUsage {
            color_attachment: true,
            .. ImageUsage::none()
        };

        let indices = Self::find_queue_families(&surface, &physical_device);

        let sharing: SharingMode = if indices.graphics_family != indices.present_family {
            vec![graphics_queue, present_queue].as_slice().into()
        } else {
            graphics_queue.into()
        };

        let (swap_chain, images) = Swapchain::new(
            device.clone(),
            surface.clone(),
            image_count,
            surface_format.0, // TODO: color space?
            extent,
            1, // layers
            image_usage,
            sharing,
            capabilities.current_transform,
            CompositeAlpha::Opaque,
            present_mode,
            FullscreenExclusive::Default, // When exclusive fullscreen is off, the app bebhaves like a borderless window
            true, // clipped
            surface_format.1
        ).expect("failed to create swap chain!");

        (swap_chain, images)
    }

    fn find_depth_format() -> Format {
        // this format is guaranteed to be supported by vulkano and as it stands now, I can't figure
        // how to do the queries performed in the original tutorial in vulkano...
        Format::D16Unorm
    }

    fn create_depth_image(device: &Arc<Device>, dimensions: [u32; 2], format: Format) -> Arc<AttachmentImage<Format>> {
        AttachmentImage::with_usage(
            device.clone(),
            dimensions,
            format,
            ImageUsage { depth_stencil_attachment: true, ..ImageUsage::none() }
        ).unwrap()
    }

    fn create_render_pass(device: &Arc<Device>, color_format: Format, depth_format: Format) -> Arc<dyn RenderPassAbstract + Send + Sync> {
        Arc::new(single_pass_renderpass!(device.clone(),
            attachments: {
                color: {
                    load: Clear,
                    store: Store,
                    format: color_format,
                    samples: 1,
                },
                depth: {
                    load: Clear,
                    store: DontCare,
                    format: depth_format,
                    samples: 1,
                    initial_layout: ImageLayout::Undefined,
                    final_layout: ImageLayout::DepthStencilAttachmentOptimal,
                }
            },
            pass: {
                color: [color],
                depth_stencil: {depth}
            }
        ).unwrap())
    }

    fn create_graphics_pipeline(
        device: &Arc<Device>,
        swap_chain_extent: [u32; 2],
        render_pass: &Arc<dyn RenderPassAbstract + Send + Sync>,
    ) -> Arc<dyn GraphicsPipelineAbstract + Send + Sync> {
        mod vertex_shader {
            vulkano_shaders::shader! {
               ty: "vertex",
               path: "src/26_shader_depthbuffering.vert"
            }
        }

        mod fragment_shader {
            vulkano_shaders::shader! {
                ty: "fragment",
                path: "src/26_shader_depthbuffering.frag"
            }
        }

        let vert_shader_module = vertex_shader::Shader::load(device.clone())
            .expect("failed to create vertex shader module!");
        let frag_shader_module = fragment_shader::Shader::load(device.clone())
            .expect("failed to create fragment shader module!");

        let dimensions = [swap_chain_extent[0] as f32, swap_chain_extent[1] as f32];
        let viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions,
            depth_range: 0.0 .. 1.0,
        };

        Arc::new(GraphicsPipeline::start()
            .vertex_input_single_buffer::<Vertex>()
            .vertex_shader(vert_shader_module.main_entry_point(), ())
            .triangle_list()
            .primitive_restart(false)
            .viewports(vec![viewport]) // NOTE: also sets scissor to cover whole viewport
            .fragment_shader(frag_shader_module.main_entry_point(), ())
            .depth_clamp(false)
            // NOTE: there's an outcommented .rasterizer_discard() in Vulkano...
            .polygon_mode_fill() // = default
            .line_width(1.0) // = default
            .cull_mode_back()
            .front_face_counter_clockwise()
            // NOTE: no depth_bias here, but on pipeline::raster::Rasterization
            .blend_pass_through() // = default
            .depth_stencil(DepthStencil::simple_depth_test())
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .build(device.clone())
            .unwrap()
        )
    }

    fn create_framebuffers(
        swap_chain_images: &[Arc<SwapchainImage<Window>>],
        render_pass: &Arc<dyn RenderPassAbstract + Send + Sync>,
        depth_image: &Arc<AttachmentImage<Format>>
    ) -> Vec<Arc<dyn FramebufferAbstract + Send + Sync>> {
        swap_chain_images.iter()
            .map(|image| {
                let fba: Arc<dyn FramebufferAbstract + Send + Sync> = Arc::new(Framebuffer::start(render_pass.clone())
                    .add(image.clone()) .unwrap()
                    .add(depth_image.clone()).unwrap()
                    .build().unwrap());
                fba
            }
        ).collect::<Vec<_>>()
    }

    fn get_mip_dim(mip_idx: u32, img_dimensions: ImageDimensions) -> Result<[i32; 3], String> {
        if let Some(dim) = img_dimensions.mipmap_dimensions(mip_idx) {
            if let ImageDimensions::Dim2d { width, height, .. } = dim {
                Ok([width as i32, height as i32, 1])
            } else {
                Err("MipMapping: Did not get 2D image for blitting".to_string())
            }
        } else {
            Err(format!("MipMapping: image has no mip map at level {}", mip_idx).to_string())
        }
    }

    fn create_texture_image(queue: &Arc<Queue>) -> Arc<ImmutableImage<Format>> {
        let image = image::open(TEXTURE_PATH).unwrap();
        let dimensions = Dimensions::Dim2d { width: image.width(), height: image.height() };

        let image_rgba = image.to_rgba();

        let image_usage = ImageUsage {
            transfer_destination: true,
            transfer_source: true,
            sampled: true,
            ..ImageUsage::none()
        };

        let (image, image_init) = ImmutableImage::uninitialized(
            queue.device().clone(),
            dimensions,
            Format::R8G8B8A8Unorm,
            MipmapsCount::Log2,
            image_usage,
            ImageLayout::TransferDstOptimal,
            queue.device().active_queue_families(),
        ).unwrap();

        let source = CpuAccessibleBuffer::from_iter(
            queue.device().clone(),
            BufferUsage { transfer_source: true, ..BufferUsage::none() },
            false, // Argument not in old version
            image_rgba.into_raw().iter().cloned()
        ).unwrap();

        let mut cb = AutoCommandBufferBuilder::new(
            queue.device().clone(), queue.family()
        ).expect("Failed to start command buffer fro image creation!");

        cb.copy_buffer_to_image_dimensions(
            source,
            image_init,
            [0, 0, 0],
            image.dimensions().width_height_depth(),
            0,
            image.dimensions().array_layers(),
            0
        ).unwrap();

        let img_dimensions = ImageAccess::dimensions(&image);

        for mip_idx in 1..image.mipmap_levels() {
            let source_dim = Self::get_mip_dim(mip_idx - 1, img_dimensions).unwrap();
            let dest_dim = Self::get_mip_dim(mip_idx, img_dimensions).unwrap();

            cb.blit_image(
                image.clone(),
                [0; 3],
                source_dim,
                0,
                mip_idx - 1,
                image.clone(),
                [0; 3],
                dest_dim,
                0,
                mip_idx,
                1,
                Filter::Linear
            ).unwrap();

            // todo -> figure out if i need to divide mipWidth and mipHeight by 2
        }

        let final_cb = cb.build().expect("failed to build MipMapping command buffer");

        let future = final_cb.execute(queue.clone()).unwrap();

        future.flush().unwrap();

        image
    }

    fn create_image_sampler(device: &Arc<Device>) -> Arc<Sampler> {
        // This is the standard sampler but for this section we are using a custom sampler to show the work we have done
        // Sampler::simple_repeat_linear(device.clone())

        let min_lod = 6.0;
        // This custom sampler used a min_lod of 6 to display the work we have done in this section
        //      changing the above variable will change which mipmap is used.
        Sampler::new(
            device.clone(),
            Filter::Linear,
            Filter::Linear,
            MipmapMode::Linear,
            SamplerAddressMode::Repeat,
            SamplerAddressMode::Repeat,
            SamplerAddressMode::Repeat,
            0.0,
            1.0,
            min_lod,
            1_000.0
        ).unwrap()
    }

    fn load_model() -> (Vec<Vertex>, Vec<u32>) {
        use tobj::{load_obj};

        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        let (models, _materials) = load_obj(MODEL_PATH, true).unwrap();

        for model in models.iter() {
            let mesh = &model.mesh;

            for index in &mesh.indices {
                let ind_usize = *index as usize;
                let pos = [
                    mesh.positions[ind_usize * 3],
                    mesh.positions[ind_usize * 3 + 1],
                    mesh.positions[ind_usize * 3 + 2],
                ];

                let color = [1.0, 1.0, 1.0];

                let tex_coord = [
                    mesh.texcoords[ind_usize * 2],
                    1.0 - mesh.texcoords[ind_usize * 2 + 1],
                ];

                let vertex = Vertex::new(pos, color, tex_coord);
                vertices.push(vertex);
                let index = indices.len() as u32;
                indices.push(index);
            }
        }

        (vertices, indices)
    }

    fn create_vertex_buffer(graphics_queue: &Arc<Queue>, vertices: Vec<Vertex>) -> Arc<dyn BufferAccess + Send + Sync> {
        let (buffer, future) = ImmutableBuffer::from_iter(
            vertices.into_iter(), BufferUsage::vertex_buffer(),
            graphics_queue.clone())
            .unwrap();
        future.flush().unwrap();
        buffer
    }

    fn create_index_buffer(graphics_queue: &Arc<Queue>, indices: Vec<u32>) -> Arc<dyn TypedBufferAccess<Content=[u32]> + Send + Sync> {
        let (buffer, future) = ImmutableBuffer::from_iter(
            indices.into_iter(), BufferUsage::index_buffer(),
            graphics_queue.clone())
            .unwrap();
        future.flush().unwrap();
        buffer
    }

    fn create_uniform_buffers(
        device: &Arc<Device>,
        num_buffers: usize,
        start_time: Instant,
        dimensions_u32: [u32; 2]
    ) -> Vec<Arc<CpuAccessibleBuffer<UniformBufferObject>>> {
        let mut buffers = Vec::new();

        let dimensions = [dimensions_u32[0] as f32, dimensions_u32[1] as f32];

        let uniform_buffer = Self::update_uniform_buffer(start_time, dimensions);

        for _ in 0..num_buffers {
            let buffer = CpuAccessibleBuffer::from_data(
                device.clone(),
                BufferUsage::uniform_buffer_transfer_destination(),
                false, // Parameter not in old version of tutorial
                uniform_buffer,
            ).unwrap();

            buffers.push(buffer);
        }

        buffers
    }

    fn create_descriptor_sets(
        device:          &Arc<Device>,
        uniform_buffers: &Vec<Arc<CpuAccessibleBuffer<UniformBufferObject>>>,
        texture_image:   &Arc<ImmutableImage<Format>>,
        image_sampler:   &Arc<Sampler>,
    ) -> Vec<DescriptorSet> {
        let descriptor_info = vec![
            Some(DescriptorDesc {
                ty: DescriptorDescTy::Buffer(
                    DescriptorBufferDesc { dynamic: Some(false), storage: false }
                ),
                array_count: 1,
                stages: ShaderStages { vertex: true, .. ShaderStages::none() },
                readonly: true,
            }),
            Some(DescriptorDesc {
                ty: DescriptorDescTy::CombinedImageSampler(DescriptorImageDesc {
                    sampled: true,
                    dimensions: DescriptorImageDescDimensions::TwoDimensional,
                    format: None,
                    multisampled: false,
                    array_layers: DescriptorImageDescArray::NonArrayed
                }),
                array_count: 1,
                stages: ShaderStages { fragment: true, .. ShaderStages::none() },
                readonly: true,
            }),
        ];

        let layout = Arc::new(UnsafeDescriptorSetLayout::new(
            device.clone(), descriptor_info.into_iter()).unwrap());
        let mut pool = FixedSizeDescriptorSetsPool::new(layout);

        uniform_buffers.iter().map(|uniform_buffer| {
            Arc::new(pool.next()
                .add_buffer(uniform_buffer.clone()).unwrap()
                .add_sampled_image(texture_image.clone(), image_sampler.clone()).unwrap()
                .build().unwrap()
            )
        }).collect()
        // No need to save the pool; it's kept alive by the allocated DescriptorSets
    }

    fn create_command_buffers(&mut self) {
        let queue_family = self.graphics_queue.family();
        let dimensions = [self.swap_chain.dimensions()[0] as f32, self.swap_chain.dimensions()[1] as f32];

        self.command_buffers = self.swap_chain_framebuffers
            .iter()
            .enumerate()
            .map(|(i, framebuffer)| {
                let mut builder = AutoCommandBufferBuilder::primary_simultaneous_use(
                    self.device.clone(), queue_family).unwrap();
                builder
                    .update_buffer(
                        self.uniform_buffers[i].clone(),
                        Self::update_uniform_buffer(self.start_time, dimensions))
                    .unwrap()
                    .begin_render_pass(framebuffer.clone(), false, vec![[0.0, 0.0, 0.0, 1.0].into(), ClearValue::Depth(1.0)])
                    .unwrap()
                    .draw_indexed(
                        self.graphics_pipeline.clone(),
                        &DynamicState::none(),
                        vec![self.vertex_buffer.clone()],
                        self.index_buffer.clone(),
                        self.descriptor_sets[i].clone(),
                        ())
                    .unwrap()
                    .end_render_pass()
                    .unwrap();
                Arc::new(builder.build().unwrap())
            }) .collect();
    }

    fn create_sync_objects(device: &Arc<Device>) -> Box<dyn GpuFuture> {
        Box::new(sync::now(device.clone())) as Box<_>
    }

    fn find_queue_families(surface: &Arc<Surface<Window>>, device: &PhysicalDevice) -> QueueFamilyIndices {
        let mut indices = QueueFamilyIndices::new();
        // TODO: replace index with id to simplify?
        for (i, queue_family) in device.queue_families().enumerate() {
            if queue_family.supports_graphics() {
                indices.graphics_family = i as i32;
            }

            if surface.is_supported(queue_family).unwrap() {
                indices.present_family = i as i32;
            }

            if indices.is_complete() {
                break;
            }
        }

        indices
    }

    fn create_logical_device(
        instance: &Arc<Instance>,
        surface: &Arc<Surface<Window>>,
        physical_device_index: usize,
    ) -> (Arc<Device>, Arc<Queue>, Arc<Queue>) {
        let physical_device = PhysicalDevice::from_index(&instance, physical_device_index).unwrap();
        let indices = Self::find_queue_families(&surface, &physical_device);

        let families = [indices.graphics_family, indices.present_family];
        use std::iter::FromIterator;
        let unique_queue_families: HashSet<&i32> = HashSet::from_iter(families.iter());

        let queue_priority = 1.0;
        let queue_families = unique_queue_families.iter().map(|i| {
            (physical_device.queue_families().nth(**i as usize).unwrap(), queue_priority)
        });

        // NOTE: the tutorial recommends passing the validation layers as well
        // for legacy reasons (if ENABLE_VALIDATION_LAYERS is true). Vulkano handles that
        // for us internally.

        let (device, mut queues) = Device::new(physical_device, &Features::none(),
            &device_extensions(), queue_families)
            .expect("failed to create logical device!");

        let graphics_queue = queues.next().unwrap();
        let present_queue = queues.next().unwrap_or_else(|| graphics_queue.clone());

        (device, graphics_queue, present_queue)
    }

    fn create_surface(instance: &Arc<Instance>, event_loop: &EventLoop<()>) -> Arc<Surface<Window>> {
        WindowBuilder::new()
            .with_title("Vulkan")
            .with_inner_size(LogicalSize::new(f64::from(WIDTH), f64::from(HEIGHT)))
            .build_vk_surface(&event_loop, instance.clone())
            .expect("failed to create window surface!")
    }

    pub fn main_loop(mut self, should_exit: Arc<RwLock<bool>>) {
        thread::Builder::new().name("Renderer".into()).spawn(move || {
            let mut last_time = Instant::now();
            let mut previous_frame_end = Some(Self::create_sync_objects(&self.device));

            while !match should_exit.read() { Ok(se) => *se, Err(_) => true } {
                while last_time.elapsed() < FRAME_PERIOD {} // Limit the framerate
                last_time = Instant::now();
                self.create_command_buffers();
                self.draw_frame(&mut previous_frame_end);
            }
        }).unwrap();
    }

    fn draw_frame(&mut self, previous_frame_end: &mut Option<Box<dyn GpuFuture>>) {
        previous_frame_end.as_mut().unwrap().cleanup_finished();

        if self.recreate_swap_chain {
            self.recreate_swap_chain();
            self.recreate_swap_chain = false;
        }

        let (image_index, acquire_future) = match acquire_next_image(self.swap_chain.clone(), None) {
            Ok((_, true, _)) | Err(AcquireError::OutOfDate) => {
                self.recreate_swap_chain = true;
                return;
            },
            Ok((index, false, future)) => (index, future),
            Err(err) => panic!("{:?}", err)
        };

        let command_buffer = self.command_buffers[image_index].clone();

        // we're joining on the previous future but the CPU is running faster than the GPU so
        // eventually it stutters, and jumps ahead to the newer frames.
        //
        // See vulkano issue 1135: https://github.com/vulkano-rs/vulkano/issues/1135
        let future = previous_frame_end.take().unwrap()
            .join(acquire_future)
            .then_execute(self.graphics_queue.clone(), command_buffer)
            .unwrap()
            .then_swapchain_present(self.present_queue.clone(), self.swap_chain.clone(), image_index)
            .then_signal_fence_and_flush();

        match future {
            Ok(future) => {
                // This makes sure the CPU stays in sync with the GPU in situations when the CPU is
                // running "too fast"
                #[cfg(target_os = "macos")]
                future.wait(None).unwrap();

                *previous_frame_end = Some(Box::new(future) as Box<_>);
            }
            Err(vulkano::sync::FlushError::OutOfDate) => {
                self.recreate_swap_chain = true;
                *previous_frame_end
                    = Some(Box::new(vulkano::sync::now(self.device.clone())) as Box<_>);
            }
            Err(e) => {
                println!("{:?}", e);
                *previous_frame_end
                    = Some(Box::new(vulkano::sync::now(self.device.clone())) as Box<_>);
            }
        }
    }

    fn update_uniform_buffer(start_time: Instant, dimensions: [f32; 2]) -> UniformBufferObject {
        let duration = Instant::now().duration_since(start_time);
        let elapsed = (duration.as_secs() * 1000) + u64::from(duration.subsec_millis());

        let model = Matrix4::from_angle_z(Rad::from(Deg(elapsed as f32 * 0.180)));

        let view = Matrix4::look_at(
            Point3::new(2.0, 2.0, 2.0),
            Point3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0)
        );

        let mut proj = cgmath::perspective(
            Rad::from(Deg(45.0)),
            dimensions[0] as f32 / dimensions[1] as f32,
            0.1,
            10.0
        );

        proj.y.y *= -1.0;

        UniformBufferObject { model, view, proj }
    }

    fn recreate_swap_chain(&mut self) {
        let dimensions = self.surface.window().inner_size().into();

        let (swap_chain, images) =
            match self.swap_chain.recreate_with_dimensions(dimensions) {
                Ok(out) => out,
                Err(e) => {
                    eprintln!("Swapchain not resized: {:?}", e);
                    return;
                }
            };

        let depth_image = Self::create_depth_image(&self.device, swap_chain.dimensions(), self.depth_format);

        self.swap_chain = swap_chain;
        self.swap_chain_images = images;

        self.render_pass = Self::create_render_pass(&self.device, self.swap_chain.format(), self.depth_format);
        self.graphics_pipeline = Self::create_graphics_pipeline(&self.device, self.swap_chain.dimensions(),
            &self.render_pass);

        self.swap_chain_framebuffers = Self::create_framebuffers(&self.swap_chain_images, &self.render_pass, &depth_image);
        self.create_command_buffers();
    }
}

struct HelloTriangleApplication {
    display: Display,
    renderer: Renderer,
}

impl HelloTriangleApplication {
    fn initialize() -> Self {
        let display = Display::new();
        let renderer = Renderer::initialize(&display);
        Self { display, renderer }
    }

    fn main_loop(self) {
        let should_exit = Arc::new(RwLock::new(false));
        self.display.run(should_exit.clone());
        self.renderer.main_loop(should_exit.clone());
    }
}

fn main() {
    let app = HelloTriangleApplication::initialize();
    app.main_loop();
}
