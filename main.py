import cv2
import numpy as np
import plotly.express as px
import pandas as pd
import skimage.exposure
import matplotlib.pyplot as plt
import os


class Config:
    # =================================================USER INPUT=======================================================
    image_path: str = 'TEM-Images/gold_nanoparticles_52x52nm.jpg'  # Path of the TEM image (square-shaped)
    image_width_real: float = 52  # Real width of the image in nm
    lattice_spacing: float = 0.25  # Lattice spacing in nm. The lower, the higher the resolution

    lambda_start: int = 350  # Lower wavenumber of the simulated range
    lambda_end: int = 800  # Upper wavenumber of the simulated range
    n_lambda: int = 46  # Number of wavenumbers in the wavenumber range
    n_ambient: float = 1.000  # Refractive index of the environment

    use_substrate: bool = False  # True or False to specify if a substrate is used
    substrate_thickness: int = 1  # Substrate thickness in units of lattice spacing

    mode_extrapolation: str = 'Kernel'  # Kernel or Extrusion method for extrapolation
    n_extrusion: int = 1  # Number of layers used for extrusion

    binary_threshold: int = 100  # Threshold for converting the image to binary
    binary_kernel_threshold: int = 100  # Threshold for converting the image to binary after applying the kernel
    particle_size_threshold: int = 500  # Particle size threshold for particle detection
    matrix_size: int = 3  # Size of the extrapolation matrix

    show_geometry: bool = True  # Flag to plot the geometry with Plotly in a browser window
    # ==================================================================================================================

    image_width: int = 0  # Image width in pixels (will be calculated later)
    image_height: int = 0  # Image height in pixels (will be calculated later)
    pixel_per_nm: float = 0.0  # Pixels per nanometer (will be calculated later)
    pixel_per_lattice_spacing: int = 0  # Pixels per lattice spacing (will be calculated later)


def setup(conf: Config) -> None:
    """
    Set up global variables based on the user input and create necessary folders.
    """
    img = cv2.imread(conf.image_path, cv2.IMREAD_COLOR)

    conf.image_width = int(np.shape(img)[0])
    conf.image_height = int(np.shape(img)[1])

    conf.pixel_per_nm = float(conf.image_width / conf.image_width_real)
    conf.pixel_per_lattice_spacing = int(conf.pixel_per_nm * conf.lattice_spacing)

    # List of folders to check and create if they do not exist
    folders = ["Scratch", "Output"]

    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)


def convert_to_binary(conf: Config) -> np.ndarray:
    """
    Convert an input image to a binary image with multiple processing steps,
    including contrast enhancement, thresholding, blurring, and resizing.

    Returns:
        np.ndarray: The processed binary image.
    """
    # Load image
    img = cv2.imread(conf.image_path)

    # Increase contrast
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl, a, b))
    enhanced_contrast_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # Save image with enhanced contrast
    cv2.imwrite("Scratch/1_enhanced_contrast.png", enhanced_contrast_image)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(enhanced_contrast_image, cv2.COLOR_BGR2GRAY)

    # Apply threshold to create a binary image
    _, binary_image = cv2.threshold(gray_image, conf.binary_threshold, 255, cv2.THRESH_BINARY)

    # Save binary image
    cv2.imwrite("Scratch/2_binary.png", binary_image)

    # Blur image
    blurred_image = cv2.GaussianBlur(binary_image, (11, 11), sigmaX=3, sigmaY=3, borderType=cv2.BORDER_DEFAULT)
    blurred_image = skimage.exposure.rescale_intensity(blurred_image, in_range=(127.5, 255), out_range=(0, 255))

    # Save blurred image
    cv2.imwrite('Scratch/3_blurred.png', blurred_image)

    # Apply kernel for smoothing
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    kernel_image = cv2.filter2D(blurred_image, -1, kernel)

    # Threshold the kernel image to binary
    kernel_image = cv2.threshold(kernel_image, conf.binary_kernel_threshold, 255, cv2.THRESH_BINARY)[1]

    # Save binary kernel image
    cv2.imwrite('Scratch/4_kernel.png', kernel_image)

    # Crop the image so the dimensions are multiples of the lattice spacing
    cropped_image = kernel_image[
                    0:conf.image_width - conf.image_width % conf.pixel_per_lattice_spacing,
                    0:conf.image_height - conf.image_height % conf.pixel_per_lattice_spacing
                    ]

    # Resize the image so every pixel represents one dipole unit
    resized_image = cv2.resize(
        cropped_image,
        None,
        fx=1 / conf.pixel_per_lattice_spacing,
        fy=1 / conf.pixel_per_lattice_spacing,
        interpolation=cv2.INTER_AREA
    ).astype("uint8")

    # Convert resized image to binary again after interpolation
    resized_image = cv2.threshold(resized_image, conf.binary_kernel_threshold, 255, cv2.THRESH_BINARY)[1]

    # Save the resized image
    cv2.imwrite("Scratch/5_resized.png", resized_image)

    return resized_image


def detect_particles(conf: Config) -> None:
    """
    Detect particles in an image, analyze their sizes and distances, and output statistics and visualizations.

    The function performs the following:
    - Detect blobs (particles) in a binary kernel image.
    - Analyze particle sizes and pairwise distances.
    - Compute and output statistics, such as the number of particles, coverage, average size, and distances.
    - Generate and save histograms of particle sizes and distances.

    Returns:
        None
    """
    # Read the grayscale kernel image from the specified path
    kernel_image = cv2.imread('Scratch/4_kernel.png', cv2.IMREAD_GRAYSCALE)

    # Convert the image to unsigned 8-bit integer format for blob detection
    img = kernel_image.astype(np.uint8)

    # Initialize parameters for blob detection
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = False  # Disable filtering by blob color
    params.filterByArea = True  # Enable filtering by blob area
    params.minArea = conf.particle_size_threshold  # Set the minimum area threshold for blobs
    params.filterByCircularity = False  # Disable filtering by blob circularity
    params.filterByConvexity = False  # Disable filtering by blob convexity
    params.filterByInertia = False  # Disable filtering by blob inertia
    params.maxArea = conf.image_width * conf.image_height  # Set the maximum area based on image dimensions
    params.filterByColor = True  # Enable filtering by blob color

    # Create a blob detector with the specified parameters
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs in the image
    keypoints = detector.detect(img)

    # Draw the detected blobs on the image
    img_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255),
                                           cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # Save the image with detected blobs highlighted
    cv2.imwrite("Scratch/6_blobs.png", img_with_keypoints)

    # Initialize arrays for particle sizes and coordinates
    particle_sizes = np.array([])  # In pixels
    particle_coordinates = np.empty([0, 2])  # Store coordinates of particles

    # Extract sizes and coordinates of detected blobs
    for blob in keypoints:
        particle_sizes = np.append(particle_sizes, blob.size)
        particle_coordinates = np.vstack([particle_coordinates, [blob.pt[0], blob.pt[1]]])

    # Compute pairwise distances between particle centers
    particle_distances = np.linalg.norm(particle_coordinates[None, :] - particle_coordinates[:, None], axis=2)

    # Set diagonal distances to infinity to exclude self-distances
    np.fill_diagonal(particle_distances, np.inf)

    # Find the minimum distance to the nearest neighbor for each particle
    min_particle_distances = np.min(particle_distances, axis=1)

    # Convert distances and sizes from pixels to real-world units (nanometers)
    min_distance_particles_real = min_particle_distances / conf.pixel_per_nm
    particle_sizes_real = particle_sizes / conf.pixel_per_nm

    # Calculate the percentage coverage of particles in the image
    coverage = np.count_nonzero(kernel_image == 0) / np.size(kernel_image) * 100

    # Output various statistics about the detected particles
    print(f'Number of detected particles: {len(particle_sizes_real)}')
    print(f'Coverage: {coverage:.1f} %')
    print(f'Average particle diameter: {np.mean(particle_sizes_real):.2f} nm')
    print(f'Median particle diameter: {np.median(particle_sizes_real):.2f} nm')
    print(f'Standard deviation particle diameter: {np.std(particle_sizes_real):.2f} nm')
    print(f'Average particle center-center distance: {np.mean(min_distance_particles_real):.2f} nm')
    print(f'Median particle center-center distance: {np.median(min_distance_particles_real):.2f} nm')
    print(f'Standard deviation particle center-center distance: {np.std(min_distance_particles_real):.2f} nm')

    # Plot and save the histogram of particle diameters
    plt.hist(particle_sizes_real, bins='auto', rwidth=0.7)
    plt.xlabel('Particle diameter (nm)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('Scratch/7_blob_size_dist.png', dpi=300)

    # Clear the current plot
    plt.clf()

    # Plot and save the histogram of particle center-to-center distances
    plt.hist(min_distance_particles_real, bins='auto', rwidth=0.75)
    plt.xlabel('Particle center-center distance (nm)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('Scratch/8_blob_distance_dist.png', dpi=300)


def create_image_stack(image: np.ndarray, conf: Config) -> np.ndarray:
    """
    Creates an image stack based on the specified extrapolation mode and user settings.

    The function supports two modes:
    1. 'Extrusion': Stacks the image on top of itself.
    2. 'Kernel': Creates additional frames based on a kernel filling mechanism.

    Parameters:
        image (np.ndarray): Input image as a 2D or 3D numpy array.
        conf (Config): The configuration object containing user input.

    Returns:
        np.ndarray: The resulting image stack as a 3D numpy array.
    """
    # Setup image stack
    image_stack = image
    identity_image = image_stack

    # Define border around the image in case parts of the matrix leave the image plane
    border_width = int((conf.matrix_size - 1) / 2)

    # Initialize kernel sum
    kernel_sum = 0

    # Create image stack based on user preference
    if conf.mode_extrapolation == 'Extrusion':
        for _ in range(conf.n_extrusion):
            # For extrusion, stack the image on top of itself
            image_stack = np.dstack((image_stack, identity_image))
    elif conf.mode_extrapolation == 'Kernel':
        # Check if there are enough black pixels to fill the matrix
        while np.count_nonzero(identity_image == 0) > conf.matrix_size ** 2 - border_width:
            # Create a border around the image
            identity_border_image = cv2.copyMakeBorder(identity_image, border_width, border_width, border_width,
                                                       border_width, cv2.BORDER_CONSTANT, value=[0, 0, 0, 0])
            # Create a new empty image for the image stack
            frame = np.full((identity_border_image.shape[0], identity_border_image.shape[1]), 255, dtype=int)
            # Iterate over the whole image
            for i in range(1, identity_border_image.shape[0] - border_width):
                for j in range(1, identity_border_image.shape[1] - border_width):
                    # Iterate through the matrix
                    for k in range(i - border_width, i + border_width + 1):
                        for m in range(j - border_width, j + border_width + 1):
                            # Calculate kernel sum
                            kernel_sum += int(identity_border_image[k, m])

                    # When the whole matrix is filled with black pixels, set a black pixel on its position and reset
                    if kernel_sum == 0:
                        frame[i, j] = 0
                    kernel_sum = 0

            # Remove the border before using the image in the image stack
            identity_image = frame[border_width:frame.shape[0] - border_width,
                                   border_width:frame.shape[1] - border_width]

            # Put the image onto the image stack
            image_stack = np.dstack((image_stack, identity_image))

    # Substrate is generated the same way as the extrusion and is placed under the image stack
    if conf.use_substrate:
        substrate_plane = np.zeros((np.shape(image)[0], np.shape(image)[1]), dtype=int)
        for _ in range(conf.substrate_thickness):
            image_stack = np.dstack((substrate_plane, image_stack))

    return image_stack


def get_image_stack_properties(image_stack: np.ndarray) -> [int, int, int, int]:
    """
    Computes the dimensions of the image stack and counts the number of black pixels (dipoles).

    In DDSCAT, the geometry plane is assumed to be the Y-Z plane.

    Parameters:
        image_stack (np.ndarray): The input 3D image stack as a numpy array.

    Returns:
        Tuple[int, int, int, int]: A tuple containing:
            - size_x (int): The size of the stack along the X-axis (depth).
            - size_y (int): The size of the stack along the Y-axis (height).
            - size_z (int): The size of the stack along the Z-axis (width).
            - n_dipoles (int): The number of black pixels in the stack.
    """
    # In DDSCAT the geometry plane is the Y-Z plane
    size_x = np.shape(image_stack)[2]  # Depth (X-axis size)
    size_y = np.shape(image_stack)[0]  # Height (Y-axis size)
    size_z = np.shape(image_stack)[1]  # Width (Z-axis size)

    # Get number of black pixels
    n_dipoles = np.count_nonzero(image_stack == 0)

    return size_x, size_y, size_z, n_dipoles


def generate_shape_file(image_stack: np.ndarray, conf: Config) -> None:
    """
    Generates a shape file (`shape.dat`) describing the 3D structure of the input image stack for DDSCAT.

    The file includes details about the geometry, lattice vectors, offsets, and material properties of dipoles.

    Parameters:
        image_stack (np.ndarray): The 3D image stack representing the shape to be described.
        conf (Config): The configuration object containing user input.

    Writes:
        Output/shape.dat: A file containing the geometry and material information for the dipoles.
    """
    # Load geometry dimensions
    size_x, size_y, size_z, n_dipoles = get_image_stack_properties(image_stack)

    # Writing the shape file
    with open('Output/shape.dat', 'w') as shape_file:
        shape_file.writelines([
            f'>TARREC   sputtered Au; AX,AY,AZ= {size_x} {size_y} {size_z}\n',
            f'{n_dipoles} = NAT\n',
            f'1.000000  0.000000  0.000000 = A_1 vector\n',
            f'0.000000  1.000000  0.000000 = A_2 vector\n',
            f'1.000000  1.000000  1.000000 = lattice spacings (d_x,d_y,d_z)/d\n',
            f'{-((size_x - 1) / 2)} {-((size_y - 1) / 2)} {-((size_z - 1) / 2)} '
            f'= lattice offset x0(1-3) = (x_TF,y_TF,z_TF)/d for dipole 0 0 0\n',
            f'    JA    IX    IY    IZ     ICOMP(x,y,z)\n'
        ])

        # Counter for indexing individual dipoles
        counter = 0

        # Set material based on the current layer and write the shape file
        for i in range(size_x):
            if conf.use_substrate and i < conf.substrate_thickness:
                material_value = 2
            else:
                material_value = 1

            image = image_stack[:, :, i]
            for j in range(size_y):
                for k in range(size_z):
                    if image[j, k] == 0:
                        counter += 1
                        shape_file.write(
                            f'{counter: >6}{i + 1: >6}{j + 1: >6}{k + 1: >6}     '
                            f'{material_value}  {material_value}  {material_value}\n')


def calculate_effective_radius(n_dipoles: int, conf: Config) -> float:
    """
    Calculates the effective radius of a system based on the number of dipoles and their spacing.

    The effective radius is determined by averaging two approaches for calculating the dipole volume:
    1. Assuming each dipole is a sphere with a diameter equal to the lattice spacing.
    2. Assuming each dipole occupies a cubic volume equivalent to the lattice spacing.

    Parameters:
        n_dipoles (int): The total number of dipoles in the system.
        conf (Config): The configuration object containing user input.

    Returns:
        float: The effective radius of the system in micrometers.
    """
    # Calculate the dipole volume (average of spherical and cubic assumptions)
    dipole_volume = ((n_dipoles * 4 / 3 * np.pi * ((0.5 * conf.lattice_spacing) ** 3) / 1E9) +
                     (n_dipoles * (conf.lattice_spacing ** 3) / 1E9)) / 2

    # Calculate the effective radius
    effective_radius = ((3 * dipole_volume) / (4 * np.pi)) ** (1 / 3)

    return effective_radius


def generate_ddscat_file(image_stack: np.ndarray, conf: Config) -> None:
    """
    Generates a DDSCAT parameter file for the specified image stack.

    This function writes the required parameters for running DDSCAT simulations,
    including information about the geometry, material properties, and scattering settings.

    Parameters:
        image_stack (np.ndarray): A 3D numpy array representing the image stack
                                  (height, width, depth), where the stack represents
                                  the geometry of the target.
        conf (Config): The configuration object containing user input.

    Returns:
        None: The function writes the DDSCAT input file ('ddscat.par') in the 'Output' directory.
    """
    # Load geometry dimensions
    size_x, size_y, size_z, n_dipoles = get_image_stack_properties(image_stack)

    # Get the effective radius
    effective_radius = calculate_effective_radius(n_dipoles, conf)

    # Writing the config file. For details consult the DDSCAT manual
    with open('Output/ddscat.par', 'w') as input_file:
        input_file.writelines([
            f"' ========== Parameter file for v7.3 ==================='\n",
            f"'**** Preliminaries ****'\n",
            f"'NOTORQ' = CMTORQ*6 (DOTORQ, NOTORQ) -- either do or skip torque calculations\n",
            f"'PBCGS2' = CMDSOL*6 (PBCGS2, PBCGST, GPBICG, QMRCCG, PETRKP) -- CCG method\n",
            f"'GPFAFT' = CMETHD*6 (GPFAFT, FFTMKL) -- FFT method\n",
            f"'GKDLDR' = CALPHA*6 (GKDLDR, LATTDR, FLTRCD) -- DDA method\n",
            f"'NOTBIN' = CBINFLAG (NOTBIN, ORIBIN, ALLBIN)\n",
            f"'**** Initial Memory Allocation ****'\n",
            f"{round(size_x * 1.1)} {round(size_y * 1.1)} {round(size_z * 1.1)}"
            f" = dimensioning allowance for target generation\n",
            f"'**** Target Geometry and Composition ****'\n",
            f"'FROM_FILE' = CSHAPE*9 shape directive\n",
            f"no SHPAR parameters needed\n"])

        if conf.use_substrate:
            input_file.writelines([
                f"2         = NCOMP = number of dielectric materials\n",
                f"'Au_Johnson_Evap' = file with refractive index 1\n",
                f"'soda-lime-glass' = file with refractive index 2\n"
            ])
        else:
            input_file.writelines([
                f"1         = NCOMP = number of dielectric materials\n",
                f"'Au_Johnson_Evap' = file with refractive index 1\n"
            ])

        input_file.writelines([
            f"'**** Additional Nearfield calculation? ****'\n",
            f"2 = NRFLD (=0 to skip nearfield calc., =1 to calculate nearfield E)\n",
            f"0.0 0.0 0.0 0.0 0.0 0.0 (fract. extens. of calc. vol. in -x,+x,-y,+y,-z,+z)\n",
            f"'**** Error Tolerance ****'\n",
            f"1.00e-5 = TOL = MAX ALLOWED (NORM OF |G>=AC|E>-ACA|X>)/(NORM OF AC|E>)\n",
            f"'**** Maximum number of iterations ****'\n",
            f"2000     = MXITER\n",
            f"'**** Integration cutoff parameter for PBC calculations ****'\n",
            f"3.00e-3 = GAMMA (1e-2 is normal, 3e-3 for greater accuracy)\n",
            f"'**** Angular resolution for calculation of <cos>, etc. ****'\n",
            f"0.5	= ETASCA (number of angles is proportional to [(3+x)/ETASCA]^2 )\n",
            f"'**** Vacuum wavelengths (micron) ****'\n",
            f"{conf.lambda_start / 1000:.3f} {conf.lambda_end / 1000:.3f} {conf.n_lambda} 'LIN'"
            f" = wavelengths (first,last,how many,how=LIN,INV,LOG)\n",
            f"'**** Refractive index of ambient medium'\n",
            f"{conf.n_ambient} = NAMBIENT\n",
            f"'**** Effective Radii (micron) **** '\n",
            f"{effective_radius:.10f} {effective_radius:.10f} 1 'LIN' = eff. radii "
            f"(1st last howmany how=LIN,INV,LOG)\n",
            f"'**** Define Incident Polarizations ****'\n",
            f"(0,0) (1.,0.) (0.,0.) = Polarization state e01 (k along x axis)\n",
            f"2 = IORTH  (=1 to do only pol. state e01; =2 to also do orth. pol. state)\n",
            f"'**** Specify which output files to write ****'\n",
            f'''0 = IWRKSC (=0 to suppress, =1 to write ".sca" file for each target orient.\n''',
            f"'**** Specify Target Rotations ****'\n",
            f"0.    0.   1  = BETAMI, BETAMX, NBETA  (beta=rotation around a1)\n",
            f"0.    90.  4  = THETMI, THETMX, NTHETA (theta=angle between a1 and k)\n",
            f"0.    180. 7  = PHIMIN, PHIMAX, NPHI (phi=rotation angle of a1 around k)\n",
            f"'**** Specify first IWAV, IRAD, IORI (normally 0 0 0) ****'\n",
            f"0   0   0    = first IWAV, first IRAD, first IORI (0 0 0 to begin fresh)\n",
            f"'**** Select Elements of S_ij Matrix to Print ****'\n",
            f"6	= NSMELTS = number of elements of S_ij to print (not more than 9)\n",
            f"11 12 21 22 31 41	= indices ij of elements to print\n",
            f"'**** Specify Scattered Directions ****'\n",
            f"'LFRAME' = CMDFRM (LFRAME, TFRAME for Lab Frame or Target Frame)\n",
            f"1 = NPLANES = number of scattering planes\n",
            f"0.  0. 180. 5 = phi, theta_min, theta_max (deg) for plane A"
        ])


def print_information(image_stack: np.ndarray) -> None:
    """
    Prints information about the image stack, including the number of dipoles
    and the estimated memory requirement for the DDSCAT calculation.

    Parameters:
    image_stack (np.ndarray): The 3D numpy array representing the image stack.

    Returns:
    None
    """
    size_x, size_y, size_z, n_dipoles = get_image_stack_properties(image_stack)
    print(f'Number of dipoles: {n_dipoles:.0f}')
    n_memory = (35 + 0.0010 * round(size_x * 1.1) * round(size_y * 1.1) * round(size_z * 1.1))
    print(f'Estimated memory requirement: {n_memory:.0f} MB')


def plot_geometry(conf: Config) -> None:
    """
    Plots a 3D scatter plot of dipoles' coordinates, differentiating between 'Particles'
    and 'Substrate' species, using a predefined discrete color palette.

    The function loads the dipole coordinates from a file, scales them, creates a DataFrame
    for easy handling, and plots them in 3D with Plotly.

    Parameters:
        conf (Config): The configuration object containing user input.

    Returns:
    None
    """
    # Load data
    dipoles_coordinates = np.loadtxt('Output/shape.dat', skiprows=7, usecols=(1, 2, 3, 4))

    # Scale coordinates
    for i in dipoles_coordinates:
        i[0] *= conf.pixel_per_lattice_spacing / conf.pixel_per_nm
        i[1] *= conf.pixel_per_lattice_spacing / conf.pixel_per_nm
        i[2] *= conf.pixel_per_lattice_spacing / conf.pixel_per_nm

    # Create DataFrame
    df = pd.DataFrame(dipoles_coordinates, columns=['x_coordinate', 'y_coordinate', 'z_coordinate', 'species'])

    # Map species to names
    species_mapping = {1: "Particles", 2: "Substrate"}
    df['species'] = df['species'].map(species_mapping)

    # Define a discrete color scale
    discrete_colors = px.colors.qualitative.Set1  # Use a predefined discrete color palette

    # Create the 3D scatter plot
    fig = px.scatter_3d(
        df,
        x='y_coordinate',
        y='z_coordinate',
        z='x_coordinate',
        color='species',
        color_discrete_sequence=discrete_colors  # Assign discrete colors
    )

    # Customize marker
    fig.update_traces(marker=dict(
        size=conf.pixel_per_nm * conf.lattice_spacing,
        line=dict(
            color='rgb(0, 0, 0)',
            width=2
        )
    ))

    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(nticks=5, range=[0, conf.image_width_real], visible=False, title='y-axis (nm)'),
            yaxis=dict(nticks=5, range=[0, conf.image_width_real], visible=False, title='z-axis (nm)'),
            zaxis=dict(nticks=5, range=[0, conf.image_width_real], visible=False, title='x-axis (nm)'),
        ),
        scene_camera=dict(eye=dict(x=0, y=0, z=1)),
        legend_title=dict(text="Species")
    )

    fig.show()


if __name__ == '__main__':
    # Create an instance of the config class
    config = Config()

    # Run the program
    setup(config)
    converted_image = convert_to_binary(config)
    detect_particles(config)
    generated_image_stack = create_image_stack(converted_image, config)
    generate_shape_file(generated_image_stack, config)
    generate_ddscat_file(generated_image_stack, config)
    print_information(generated_image_stack)

    # Show geometry
    if config.show_geometry:
        plot_geometry(config)
