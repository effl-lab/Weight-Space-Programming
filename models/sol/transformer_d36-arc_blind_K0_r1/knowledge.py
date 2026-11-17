"""Hand-authored science associations. No corpus, model, or fitted parameters.

Each row is a small conceptual neighborhood, not an example question.  A word
can belong to several neighborhoods.  The build uses fixed random-feature
codes to superpose these neighborhoods in the narrow architecture.
"""

from morphology import aliases_for

# topic | concept name | associated words (underscore is not a token separator)
GROUPS = r'''
life | living | living alive organisms organism grow grows reproduce reproduction cells cellular
life | cell | cell cells membrane cytoplasm nucleus organelles microscopic
life | nucleus | nucleus nuclei dna chromosomes genes genetic heredity control controls
life | mitochondria | mitochondria mitochondrion respiration energy cellular atp
life | chloroplast | chloroplast chloroplasts chlorophyll green photosynthesis sunlight plants
life | membrane | membrane membranes cell cells selectively permeable transport boundary
life | cellwall | wall walls cellulose rigid support plant plants cell
life | vacuole | vacuole vacuoles storage stores water cell plants
life | organization | cells tissues tissue organs organ systems organism organization
life | microscope | microscope microscopes magnify magnification microscopic cells lens lenses
life | unicellular | unicellular single bacteria bacterium amoeba paramecium protist
life | bacteria | bacteria bacterium microorganisms microscopic decomposers decomposition
life | viruses | virus viruses host infection infectious disease cells reproduce
life | photosynthesis | photosynthesis plants plant light sunlight chlorophyll glucose sugar food oxygen
life | photosynthesis_inputs | photosynthesis dioxide carbon water sunlight producers
life | respiration | respiration oxygen glucose sugar energy mitochondria dioxide cellular
life | roots | roots root absorb absorbs water minerals nutrients soil anchorage
life | leaves | leaves leaf sunlight photosynthesis chlorophyll stomata gas gases
life | stems | stem stems transport support water xylem phloem
life | flowers | flower flowers flowering petals pollen pollination reproduction seeds
life | pollination | pollen pollination flowers bees insects wind fertilization
life | seeds | seed seeds germination germinate embryo water warmth sprout
life | fruit | fruit fruits seeds seed dispersal animals
life | transpiration | transpiration evaporation leaves stomata water vapor
life | tropism | tropism phototropism plants light grow toward stimulus
life | producers | producer producers plants algae grass photosynthesis sunlight autotroph
life | consumers | consumer consumers animals eat eating food heterotroph
life | herbivores | herbivore herbivores rabbit rabbits deer cow cows grass plants leaves
life | carnivores | carnivore carnivores predators predator lions lion wolves wolf hawk hawks meat
life | omnivores | omnivore omnivores bear bears humans raccoon plants animals
life | decomposers | decomposer decomposers bacteria fungi fungus mushrooms dead decay recycling nutrients
life | foodchain | food chain chains web webs energy feeding consumers producers
life | energyflow | sun sunlight producers plants consumers food energy chain
life | predator | predator predators prey hunt hunting capture eat
life | prey | prey predators escape protection hide camouflage survive survival
life | camouflage | camouflage color coloration blend blending hide hiding predators environment
life | mimicry | mimicry resemble resemblance appearance protection predators poisonous
life | adaptation | adaptation adaptations traits survival survive environment habitat
life | insulation | fur feathers blubber fat insulation warmth warm cold mammals
life | migration | migration migrate birds seasonal seasons winter warmer food
life | hibernation | hibernation hibernate winter cold dormant conserve energy
life | desertlife | cactus cacti spines waxy desert conserve water drought
life | aquatic | fish fishes gills fins swimming swim aquatic dissolved oxygen water
life | birds | bird birds feathers wings beaks eggs flight fly
life | mammals | mammal mammals fur hair milk live offspring warm
life | reptiles | reptile reptiles scales eggs snake snakes lizard turtle cold
life | amphibians | amphibian amphibians frogs frog toads tadpoles moist skin metamorphosis
life | insects | insect insects six legs exoskeleton antennae thorax abdomen
life | metamorphosis | metamorphosis butterfly butterflies caterpillar larva pupa tadpole frog
life | vertebrates | vertebrate vertebrates backbone spine skeleton bones
life | invertebrates | invertebrate invertebrates insects worms jellyfish exoskeleton
life | classification | classification classify characteristics species genus kingdom taxonomy related
life | fossils_evolution | fossils fossil evolution ancestors ancestry extinct organisms evidence
life | selection | natural selection variation variations inherited traits survival reproduce environment
life | heredity | heredity inherited genetic genes gene dna chromosomes parents offspring traits
life | dominant | dominant allele alleles recessive inherited genetic trait
life | acquired | acquired learned learning practice scars experiences behavior
life | asexual | asexual identical offspring one parent cloning binary fission
life | sexual | sexual reproduction two parents offspring variation sperm egg fertilization
life | reproduction | reproduction reproduce offspring young species survival
life | competition | competition compete limited resources food water space sunlight
life | carrying_capacity | population populations carrying capacity resources limiting limited
life | biodiversity | biodiversity diversity variety species ecosystem ecosystems stability
life | habitat | habitat habitats shelter food water space organisms environment
life | ecosystem | ecosystem ecosystems organisms living nonliving interaction interactions
life | mutualism | mutualism mutualistic both benefit beneficial symbiosis bees flowers
life | parasitism | parasite parasites parasitism host harm harmful ticks fleas
life | commensalism | commensalism benefit unaffected symbiosis
life | nutrientcycle | nutrients nutrient recycling decomposers soil nitrogen cycle
life | nitrogencycle | nitrogen bacteria soil fixation nitrates legumes protein
life | carboncycle | carbon dioxide atmosphere photosynthesis respiration combustion
life | digestion | digestion digestive stomach intestines mouth nutrients food breakdown
life | absorption | small intestine intestines nutrients absorption bloodstream
life | circulation | circulatory circulation heart blood vessels arteries veins capillaries oxygen
life | heart | heart pumps pump blood circulatory muscle
life | respiratory | respiratory lungs lung breathing breath oxygen dioxide exchange
life | excretory | excretory kidney kidneys urine urea waste wastes filter filtration
life | nervous | nervous nerves nerve brain spinal cord signals impulses stimulus response
life | endocrine | endocrine hormones hormone glands gland chemical messages messenger blood
life | immune | immune immunity antibodies antibody disease pathogens infection defense vaccination
life | lymphatic | lymphatic lymph nodes immune infection defense
life | integumentary | integumentary skin protective protection barrier sweat
life | skeletal | skeletal skeleton bones support protection joints
life | muscular | muscular muscles muscle movement contract contraction bones
life | homeostasis | homeostasis temperature stable balance sweating shivering regulation
life | nutrition | nutrition nutrients balanced diet vitamins minerals proteins carbohydrates fats
life | protein | protein proteins growth repair amino acids tissue
life | carbohydrates | carbohydrates carbohydrate starch sugar sugars glucose energy
life | waterhealth | water hydration dehydration drink fluids sweat
life | exercise | exercise activity muscles heart breathing oxygen healthy health
life | vision | eye eyes vision retina light optic lens
life | hearing | ear ears hearing sound eardrum vibrations cochlea
life | touch | skin touch pressure receptors nervous stimulus
matter | atoms | atom atoms nucleus proton protons neutron neutrons electron electrons
matter | atomic_number | atomic number protons proton element identity
matter | atomic_mass | atomic mass protons neutrons nucleus isotopes
matter | electrons | electron electrons negative negatively charge cloud outer shell
matter | protons | proton protons positive positively nucleus charge
matter | neutrons | neutron neutrons neutral nucleus isotopes
matter | elements | element elements atom atoms periodic table substance
matter | molecules | molecule molecules atoms chemically bonded compounds bonds
matter | compounds | compound compounds elements chemically combined fixed ratio molecule
matter | mixtures | mixture mixtures substances physically separate separation
matter | solutions | solution solutions dissolve dissolved solute solvent homogeneous
matter | solvent | solvent solvents water dissolve dissolving liquid
matter | solubility | solubility soluble dissolve dissolves temperature stirring smaller particles
matter | filtration | filtration filter filtering insoluble solid liquid sand water
matter | evaporation_separation | evaporation evaporate salt saltwater solution separate dissolved
matter | distillation | distillation boiling evaporation condensation separate liquids temperatures
matter | magnet_separation | magnet magnetic iron filings separate separation
matter | chromatography | chromatography pigments ink colors separation
matter | covalent | covalent share shared sharing electron electrons bonds molecules
matter | oxygenbond | oxygen o2 covalent double two bonds diatomic
matter | ionic | ionic transfer transferred electrons ions charged sodium chloride salt
matter | metals | metal metals metallic conduct conductors conductivity shiny malleable ductile
matter | nonmetals | nonmetal nonmetals brittle dull insulators poor conductivity
matter | solids | solid solids definite fixed shape volume tightly particles vibrate
matter | liquids | liquid liquids definite volume container flow flowing shape
matter | gases | gas gases gaseous expand compressible container volume particles apart
matter | plasma | plasma ionized ions charged stars lightning
matter | melting | melting melt melts solid liquid heat heating warmer
matter | freezing | freezing freeze freezes liquid solid cooling cold ice
matter | evaporation | evaporation evaporate evaporates liquid gas vapor heat surface
matter | boiling | boiling boil bubbles liquid gas vapor temperature
matter | condensation | condensation condense condenses gas vapor liquid droplets cool cooling
matter | sublimation | sublimation sublime solid gas dry ice
matter | deposition | deposition gas solid frost vapor freezing
matter | physical_change | physical change changes melting freezing cutting dissolving state shape
matter | chemical_change | chemical change reaction reactions new substance rust burning gas color
matter | rust | rust rusting oxidation iron oxygen water chemical
matter | combustion | combustion burning burn oxygen fuel heat light chemical
matter | conservation_mass | conservation mass matter conserved reactants products chemical reaction closed total created destroyed
matter | reaction | reaction reactions reactants products chemical bonds atoms rearranged
matter | acid | acid acids acidic sour ph low hydrogen vinegar lemon
matter | base | base bases basic alkaline bitter slippery ph hydroxide soap
matter | neutralization | neutralization acid base salt water neutral seven
matter | density | density dense mass volume sink sinks sinking compact
matter | buoyancy | buoyancy buoyant float floats floating less dense displaced water
matter | gas_pressure | pressure gas gases particles collisions container compressed
matter | thermal_expansion | expansion expand expands heating heated temperature particles apart
matter | contraction | contract contraction contracts cooling colder particles closer
matter | water_expansion | ice water freezing expands less dense floats
matter | saltwater | saltwater ocean salt dissolved saline salinity
matter | hydrocarbons | hydrocarbon hydrocarbons carbon hydrogen fossil fuels petroleum
matter | organic | organic carbon compounds living organisms
matter | periodic | periodic table groups periods elements properties atomic
matter | conductors | conductor conductors copper aluminum aluminium silver metal metals electrical
matter | insulators | insulator insulators rubber plastic glass wood air insulation
energy | energy | energy work motion heat light electrical chemical transfer transformation
energy | kinetic | kinetic motion moving speed velocity mass energy
energy | potential | potential stored height elevated stretched spring gravity energy
energy | gravitational_potential | height higher raised lifted gravitational potential stored
energy | chemical_energy | chemical food fuel fuels battery batteries stored energy bonds
energy | electrical_energy | electricity electrical electrons current wires circuit energy
energy | radiant | radiant radiation light electromagnetic solar sunlight energy
energy | heat | heat thermal temperature particles motion kinetic warmer colder
energy | heatflow | heat flows transfer warmer hotter cooler colder thermal
energy | conduction | conduction contact touching metal spoon solid heat transfer
energy | convection | convection currents fluids air water rise rising sink sinking density
energy | radiation | radiation radiant sunlight sun space vacuum electromagnetic waves
energy | insulation | insulation insulator insulated trapped air foam wool reduce heat
energy | temperature | temperature thermometer degrees celsius fahrenheit kelvin measure
energy | friction | friction rubbing rough surfaces heat resistance motion slows
energy | conservation | conservation conserved energy transformed transferred created destroyed total
energy | renewables | renewable renewables solar wind hydroelectric geothermal biomass sunlight
energy | nonrenewables | nonrenewable nonrenewables coal oil petroleum gasoline gas fossil limited
energy | solar | solar sun sunlight panels cells electricity renewable
energy | wind | wind turbines turbine electricity renewable moving air
energy | hydroelectric | hydroelectric hydropower dam dams flowing water electricity renewable
energy | geothermal | geothermal earth underground heat renewable
energy | biomass | biomass wood crops organic burning renewable biofuel
energy | fossilfuels | fossil fuels coal oil petroleum gas ancient remains nonrenewable
energy | pollution | pollution pollutants burning fossil fuels emissions smog air
energy | greenhouse | greenhouse warming dioxide carbon methane atmosphere trapped heat global
energy | conservation_practice | conserve conservation save energy efficient efficiency insulation recycling
motion | force | force forces push pull motion acceleration direction
motion | newton_inertia | inertia rest motion object straight balanced unbalanced
motion | newton_acceleration | acceleration force mass greater less newton
motion | action_reaction | action reaction equal opposite forces rocket thrust
motion | balanced | balanced equal opposite forces rest constant velocity equilibrium
motion | unbalanced | unbalanced net force acceleration speed direction change
motion | gravity | gravity gravitational attraction mass masses distance falling weight
motion | weight | weight gravity gravitational force newtons mass planet
motion | mass | mass matter amount grams kilograms balance inertia
motion | speed | speed distance time faster slower rate traveled
motion | velocity | velocity speed direction motion
motion | acceleration | acceleration change velocity speed direction time
motion | momentum | momentum mass velocity collision moving
motion | friction | friction surfaces rough resistance rubbing heat slow
motion | simple_machines | machines machine work force distance mechanical advantage
motion | lever | lever fulcrum pivot load effort seesaw balance
motion | pulley | pulley pulleys rope wheel lift direction force
motion | inclined_plane | inclined plane ramp slope distance force work
motion | screw | screw inclined plane spiral
motion | wedge | wedge knife axe splitting cutting inclined
motion | wheel_axle | wheel axle wheels rotate rotation machines
motion | work | work force distance joules energy movement
motion | magnetism | magnetic magnet magnets iron nickel cobalt attract repel poles
motion | magnetic_poles | north south opposite attract like repel poles magnets
motion | electromagnet | electromagnet coil wire current iron magnetic increase turns
motion | current | current electrons flow electric circuit closed amperes
motion | voltage | voltage volt potential difference battery batteries electric
motion | resistance | resistance resistor ohms wire current length thickness
motion | closed_circuit | closed complete circuit current electricity flow bulb lights
motion | open_circuit | open broken circuit switch gap current stops
motion | series | series circuit one path bulbs current
motion | parallel | parallel circuit separate multiple paths branches bulbs
motion | static | static electricity charge charges electrons rubbing attract repel
motion | waves | wave waves energy medium disturbance vibration frequency wavelength
motion | transverse | transverse perpendicular wave particles oscillation
motion | longitudinal | longitudinal parallel compression compressions rarefaction sound
motion | frequency | frequency vibrations cycles hertz pitch higher shorter
motion | wavelength | wavelength crest crests trough distance frequency shorter
motion | amplitude | amplitude height loudness louder energy sound wave
motion | sound | sound vibrations vibrating medium air hearing longitudinal
motion | vacuum_sound | vacuum sound medium particles travel space
motion | light | light electromagnetic vacuum straight rays visible
motion | reflection | reflection reflect reflects mirror shiny bounce bouncing angle
motion | refraction | refraction refract refracts bending bends lens water glass speed
motion | absorption | absorption absorb absorbs dark black heat light
motion | transparent | transparent clear transmit transmits light glass
motion | opaque | opaque block blocks light shadow shadows
motion | translucent | translucent scattered some light frosted
motion | prism | prism rainbow spectrum colors dispersion white light
motion | color | color colors reflected reflection wavelength visible
motion | lenses | lens lenses convex concave focus magnify refraction
motion | pitch | pitch frequency high low sound vibration
motion | loudness | loudness amplitude loud louder soft sound energy
motion | echo | echo echoes reflection sound
motion | doppler | doppler pitch frequency approaching receding source
motion | spectrum | electromagnetic spectrum radio microwave infrared visible ultraviolet rays gamma
motion | infrared | infrared thermal heat radiation
motion | ultraviolet | ultraviolet sun skin damage sunscreen
motion | radio | radio waves communication long wavelength
motion | xray | x rays bones medical short wavelength
space | solar_system | solar system sun planets orbit gravity
space | sun | sun star stars light heat energy nuclear fusion hydrogen
space | earth | earth planet terrestrial rock water life atmosphere
space | moon | moon satellite lunar orbit earth reflected sunlight
space | rotation | rotation rotate rotates spinning axis day night daily
space | revolution | revolution revolve orbit orbits year annual sun
space | seasons | seasons seasonal tilt tilted axis sunlight angle summer winter
space | summer | summer warmer longer days direct sunlight hemisphere
space | winter | winter colder shorter days indirect sunlight hemisphere
space | sunrise | sunrise rises rise east eastern sun dawn
space | sunset | sunset sets set west western sun evening
space | shadows | shadow shadows sun position angle noon shortest
space | moon_phases | phases moon lunar reflected sun positions waxing waning
space | newmoon | new moon sun earth same side invisible
space | fullmoon | full moon opposite earth sun illuminated
space | eclipse_solar | solar eclipse moon between sun earth shadow
space | eclipse_lunar | lunar eclipse earth between sun moon shadow
space | tides | tides tidal moon gravity sun oceans high low
space | planets_inner | mercury venus earth mars rocky terrestrial inner solid
space | planets_outer | jupiter saturn uranus neptune gaseous giant outer gas
space | mercury | mercury closest nearest sun small planet
space | venus | venus hottest thick atmosphere dioxide carbon
space | mars | mars red iron rocky planet
space | jupiter | jupiter largest giant planet gas
space | saturn | saturn rings ring gas planet
space | asteroids | asteroid asteroids rocky belt mars jupiter
space | comets | comet comets ice dust tail sun orbit
space | meteors | meteor meteors meteorite meteorites atmosphere burning rock
space | stars | star stars fusion hydrogen helium light temperature
space | galaxy | galaxy galaxies milky way stars billions
space | universe | universe galaxies expansion big bang
space | star_color | blue hotter hottest red cooler coolest stars temperature
space | telescope | telescope telescopes distant space stars planets lens mirror
space | gravity_orbit | gravity gravitational orbit attraction mass distance satellite
space | distance | light year lightyear astronomical distance stars space
space | fusion | fusion nuclear hydrogen helium sun stars energy
space | earth_gravity | earth gravity center downward fall falling
land | earth_layers | crust mantle core layers earth inner outer
land | core | core iron nickel center hottest liquid outer solid inner
land | mantle | mantle convection magma rock heat currents plates
land | crust | crust surface thin solid rocks continents ocean
land | plate_tectonics | plate plates tectonic tectonics moving boundaries continents
land | earthquakes | earthquake earthquakes fault faults plates sudden movement vibrations seismic
land | volcanoes | volcano volcanoes volcanic magma lava eruption molten
land | mountains | mountains mountain collision convergent uplift plates
land | divergent | divergent apart separating plates ridge seafloor spreading new crust
land | convergent | convergent collision collide subduction trench mountains plates
land | transform | transform sliding slide past fault earthquake
land | igneous | igneous magma lava cooling cooled solidification crystals
land | sedimentary | sedimentary sediment sediments layers compaction cementation fossils
land | metamorphic | metamorphic heat pressure changed rock rocks
land | rockcycle | rock cycle weathering erosion melting cooling heat pressure
land | minerals | mineral minerals inorganic crystal crystals hardness streak luster cleavage
land | weathering | weathering breaking breakdown smaller rocks rock chemical physical
land | frost_weathering | freezing ice cracks crack pavement expansion weathering
land | erosion | erosion erode transport carried carrying water wind glaciers sediments
land | deposition | deposition deposited sediment settle settling slower delta
land | soil | soil humus minerals weathering organic decayed rocks particles
land | humus | humus decayed decomposed organic nutrients soil
land | soil_conservation | soil conservation roots plants vegetation prevent erosion terraces
land | sand | sand sandy large particles drainage drains permeability
land | clay | clay small tiny particles holds water impermeable
land | loam | loam soil sand silt clay humus fertile
land | fossils | fossil fossils sedimentary remains organisms past extinct evidence
land | geologic_time | layers oldest older bottom younger top fossils sedimentary
land | index_fossil | index fossil fossils widespread short time relative age
land | radioactive | radioactive decay dating half life absolute age
land | watercycle | water cycle evaporation condensation precipitation runoff collection
land | clouds | clouds cloud condensation vapor droplets cool atmosphere
land | precipitation | precipitation rain snow sleet hail water clouds
land | runoff | runoff surface water streams rivers downhill
land | infiltration | infiltration infiltrate groundwater soil seep percolation
land | groundwater | groundwater aquifer aquifers wells porous permeable underground
land | watershed | watershed drainage basin rivers streams runoff
land | freshwater | freshwater fresh lakes rivers glaciers ice drinking
land | oceans | ocean oceans salt saltwater most water earth
land | glaciers | glacier glaciers ice frozen freshwater erosion movement
land | evaporation_water | evaporation sunlight solar heat liquid vapor atmosphere
land | humidity | humidity water vapor moisture air atmosphere
land | weather | weather temperature precipitation wind humidity air pressure daily
land | climate | climate average weather long years patterns region
land | air_pressure | air pressure barometer atmosphere altitude lower higher
land | wind | wind air pressure differences high low movement
land | sea_breeze | breeze land sea uneven heating convection wind
land | warm_air | warm air rising rises lower density convection
land | cold_air | cold air sinking sinks denser pressure
land | fronts | front fronts air masses warm cold weather storms
land | low_pressure | low pressure rising air clouds cloudy precipitation storm
land | high_pressure | high pressure sinking air clear dry fair
land | storms | storms storm hurricane tornado thunderstorm wind lightning
land | hurricane | hurricane hurricanes warm ocean water moist tropical
land | tornado | tornado tornadoes funnel rotating wind thunderstorm
land | lightning | lightning electrical discharge charges thunder
land | thunder | thunder sound lightning heated air expansion
land | instruments | thermometer temperature barometer pressure anemometer wind hygrometer humidity
land | rain_gauge | rain gauge precipitation rainfall measure
land | anemometer | anemometer wind speed measure
land | windvane | vane wind direction measure
land | atmosphere | atmosphere air nitrogen oxygen gases earth
land | ozone | ozone ultraviolet stratosphere radiation protects
land | troposphere | troposphere weather clouds lowest atmosphere
land | water_pollution | pollution pollutants runoff fertilizer sewage water contamination
land | acid_rain | acid rain sulfur dioxide nitrogen pollution fossil fuels
land | global_warming | warming greenhouse carbon dioxide methane fossil fuels temperature climate
land | recycling | recycling recycle reuse reduce resources waste conservation
land | deforestation | deforestation trees forest habitat loss erosion dioxide carbon
land | resources | natural resources water minerals soil forests fuels
method | experiment | experiment experiments investigation test hypothesis evidence variables
method | hypothesis | hypothesis hypotheses prediction explanation testable tested
method | evidence | evidence data observations results support conclusion
method | observation | observation observe senses see smell hear measure
method | inference | inference infer interpretation explanation observations evidence
method | conclusion | conclusion conclusions results evidence data supported
method | independent | independent variable changed manipulated tested factor
method | dependent | dependent variable measured responding response result
method | control | control controlled constant constants same comparison fair
method | repeated | repeat repeated trials multiple reliable reliability accuracy
method | sample | sample samples random representative population larger many
method | measurement | measurement measure units instruments ruler scale balance thermometer
method | mass_measure | balance scale mass grams kilograms measure weigh weighing
method | volume_measure | graduated cylinder volume milliliters liters liquid measure
method | distance_measure | ruler meterstick length millimeters centimeters meters kilometers distance measure
method | time_measure | stopwatch clock seconds minutes time measure
method | temperature_measure | thermometer degrees celsius temperature measure
method | graph | graph graphs data table tables trends relationship patterns
method | model | model models representation represent simplified diagram simulation
method | safety | safety safe gloves goggles protective washing hands clean laboratory
method | unsafe | unsafe dangerous hazard avoid eating drinking food chemicals laboratory
method | accident | accident spill spills notify teacher adult immediately
method | fairness | fair experiment same controlled constants variables one changed
method | technology | technology tools invention devices improve useful solve problems
method | communication | communication share results scientists published repeat evidence
method | engineering | engineering design problem solution test improve constraints
method | units | metric meters grams liters seconds standard units measurement
life | vascular_transport | vascular xylem roots stems water minerals transport
life | phloem_transport | phloem sugars food leaves stems roots transport
life | stomatal_exchange | stomata stomatal pores leaves exchange gases transpiration
life | photoperiod | photoperiod flowering daylight hours seasons plants
life | plant_storage | potato potatoes tuber bulbs carrots roots stored starch
life | seed_dispersal | dispersal seeds wind hooks wings floating animals fruit
life | spores | spores spore fungi mosses ferns reproduction
life | conifers | conifer conifers pine cones needles evergreen seeds
life | deciduous | deciduous oak maple leaves autumn fall seasonal
life | grassland | grassland grasslands prairie grasses herbivores grazing soil
life | rainforest | rainforest tropical warm rain rainfall biodiversity forest
life | tundra | tundra permafrost cold frozen mosses lichens
life | estuary | estuary estuaries freshwater saltwater brackish river ocean
life | wetland | wetland wetlands marsh swamp filter flooding habitat
life | coral_reef | coral reef reefs warm shallow marine biodiversity
life | scavengers | scavenger scavengers vulture vultures dead carcasses animals
life | detritivores | earthworm earthworms worm worms soil detritus decomposing organic
life | symbiosis_lichen | lichen lichens fungus algae mutualism
life | pollinator_mutualism | bee bees nectar flowers pollen mutualism
life | parasite_examples | tapeworm tick ticks flea fleas mosquito mosquitoes blood host parasites
life | keystone_species | keystone species ecosystem predators balance population
life | habitat_fragmentation | fragmentation roads construction habitat populations isolated
life | invasive_species | invasive introduced nonnative species competition native resources
life | succession | succession pioneer lichens grasses shrubs trees communities
life | primary_succession | primary succession bare rock lichens volcano lava
life | secondary_succession | secondary succession soil fire forest abandoned field
life | oxygen_consumption | respiration breathing exercising oxygen muscles energy
life | diaphragm | diaphragm lungs breathing contracts expands chest inhalation
life | alveoli | alveoli lungs capillaries oxygen diffusion exchange
life | red_blood_cells | red blood hemoglobin oxygen transport cells
life | white_blood_cells | white blood immune cells fight infection pathogens
life | platelets | platelets blood clot clotting bleeding wound
life | arteries | arteries artery heart blood away vessels
life | veins | veins vein heart blood return valves
life | villi | villi intestine nutrients absorption surface area
life | enzymes | enzyme enzymes catalyst catalysts digestion chemical reaction proteins
life | salivary_amylase | saliva mouth amylase starch digestion sugar
life | liver | liver bile detoxification digestion fats
life | pancreas | pancreas insulin glucagon blood sugar endocrine
life | insulin | insulin diabetes glucose blood sugar pancreas hormones
life | sweating | sweating sweat evaporation cooling temperature skin
life | shivering | shivering muscle contractions heat cold temperature
life | tendon | tendon tendons muscles bones attachment
life | ligament | ligament ligaments bone bones joint joints
life | cartilage | cartilage cushion joints friction bones
life | reflex | reflex reflexes automatic rapid spinal response stimulus
life | cerebrum | cerebrum brain thinking memory learning voluntary
life | cerebellum | cerebellum brain balance coordination movement
life | sensory_organs | senses sensory receptors eyes ears nose tongue skin
life | smell | nose smell olfactory chemicals receptors
life | taste | tongue taste buds sweet salty sour bitter
life | pupil | pupil iris eye light opening regulates
life | rna | rna ribonucleic ribosome ribosomes protein synthesis
life | dna_structure | dna deoxyribonucleic double helix nucleotide genes chromosomes
life | cell_division | mitosis division cells growth repair identical chromosomes
life | meiosis | meiosis gametes sperm egg half chromosomes sexual
life | diffusion | diffusion concentration particles high low movement membrane
life | osmosis | osmosis water membrane concentration selectively permeable
life | active_transport | active transport energy membrane low high concentration
life | food_pyramid | trophic pyramid energy producers consumers less lost heat
life | photosynthesis_products | photosynthesis glucose sugar oxygen products chloroplast
life | cellular_waste | respiration dioxide carbon water waste products
life | tetrapods | amphibians reptiles mammals four 4 limbs
life | insect_legs | insect insects six 6 legs
life | spider_legs | spider spiders arachnid arachnids eight 8 legs
life | insect_body | insects head thorax abdomen three 3 segments
life | fish_temperature | fish reptiles amphibians ectothermic cold blooded environment
life | mammal_temperature | mammals birds endothermic warm blooded constant temperature
life | behavioral_adaptation | behavior behavioral nocturnal migration hibernation learned instinct
life | nocturnal | nocturnal night owl owls bats active dark
life | drought_adaptation | drought cactus succulent roots waxy water storage desert
life | predator_features | predator predators claws talons fangs teeth hunting carnivore
life | herbivore_teeth | herbivore flat grinding molars teeth plants
life | disease_spread | contagious infectious spread bacteria virus germs contact
life | sanitation | sanitation hygiene wash soap clean hands germs disease
life | food_preservation | refrigeration freezing cold food bacteria spoilage slower
life | antibiotics | antibiotics antibiotic bacteria bacterial infection medicine
life | vaccination | vaccination vaccine vaccines immune antibodies prevention immunity
life | vitamin_c | vitamin scurvy citrus fruit nutrition
life | calcium | calcium bones teeth milk minerals strong
life | iron_nutrition | iron hemoglobin blood anemia nutrient
matter | carbon_atom | carbon 6 six protons atomic number
matter | oxygen_atom | oxygen 8 eight protons atomic number
matter | hydrogen_atom | hydrogen 1 one proton atomic number
matter | helium_atom | helium 2 two protons atomic number
matter | nitrogen_atom | nitrogen 7 seven protons atomic number
matter | isotope_carbon | isotope isotopes carbon radioactive neutrons dating
matter | water_formula | water h2o hydrogen oxygen two 2 one 1 molecule
matter | dioxide_formula | dioxide co2 carbon oxygen two 2 one 1 molecule
matter | salt_formula | salt sodium chloride nacl ionic compound
matter | air_mixture | air mixture nitrogen oxygen gases atmosphere
matter | alloy | alloy alloys brass bronze steel mixture metals
matter | suspension | suspension heterogeneous particles settle mixture
matter | colloid | colloid colloids dispersed milk fog particles mixture
matter | saturation | saturated saturation solution solubility dissolve maximum
matter | concentration | concentration concentrated dilute solute solution solvent
matter | stirring | stirring stir agitation dissolve faster dissolving
matter | surface_area | surface area smaller crushed powdered particles faster reaction dissolve
matter | catalysts | catalyst catalysts enzyme enzymes speed reaction unchanged
matter | exothermic | exothermic release releases heat energy surroundings warmer
matter | endothermic | endothermic absorb absorbs heat energy surroundings cooler
matter | combustion_products | combustion burning fuel oxygen dioxide carbon water products
matter | oxidation | oxidation rust rusting oxygen iron chemical reaction
matter | precipitate | precipitate precipitation insoluble solid solution reaction
matter | boiling_water | water boiling boil 100 celsius 212 fahrenheit steam
matter | freezing_water | water freezing freeze ice 0 celsius 32 fahrenheit
matter | neutral_ph | neutral ph 7 seven pure water
matter | dense_metals | lead iron steel copper metal dense sinks
matter | low_density | cork wood foam float low density
matter | gas_compression | gas gases compress pressure smaller volume particles closer
matter | gas_diffusion | gas gases odor perfume diffusion spreading particles motion
matter | brownian | brownian random motion particles molecules collision
matter | metal_properties | malleable hammered sheets ductile wire metals conductive
matter | copper | copper wire wires electrical thermal conductor metal
matter | aluminum | aluminum aluminium metal light conductor cans foil
matter | rubber | rubber electrical insulator gloves soles plastic
matter | glass | glass transparent brittle sand silica insulator
matter | noble_gases | noble gases helium neon argon unreactive stable outer shell
matter | halogens | halogen halogens chlorine fluorine reactive nonmetal
matter | alkali_metals | alkali metals sodium potassium lithium reactive
energy | thermal_equilibrium | equilibrium thermal same temperature heat transfer stops
energy | heat_capacity | specific heat capacity water temperature slowly energy
energy | latent_heat | melting boiling phase change constant temperature energy heat
energy | dark_surface | dark black absorption absorb sunlight warmer heat
energy | light_surface | white shiny reflection reflect light cooler
energy | heat_conducting | copper aluminum iron metals heat conduction conductor
energy | heat_insulating | foam wool air feathers fur insulation trapped heat
energy | thermos | thermos vacuum insulated reduce conduction convection radiation
energy | efficiency | efficiency efficient useful output input wasted heat energy
energy | electric_motor | motor electrical mechanical motion energy transformation
energy | generator | generator mechanical electrical motion energy transformation
energy | battery | battery batteries chemical electrical stored energy
energy | flashlight | flashlight battery electrical chemical light energy
energy | solar_cell | photovoltaic panel solar sunlight light electrical energy
energy | photosynthesis_energy | photosynthesis sunlight light chemical food glucose energy
motion | distance_formula | speed distance time divided ratio rate
motion | force_units | force newton newtons n spring scale
motion | energy_units | energy work joule joules j
motion | power | power watt watts energy time rate
motion | acceleration_units | acceleration meters second squared velocity time
motion | gravity_distance | gravity gravitational distance farther weaker closer stronger
motion | gravity_mass | gravity gravitational mass massive larger stronger
motion | terminal_velocity | terminal velocity air resistance gravity balanced constant falling
motion | air_resistance | air resistance drag parachute surface area falling slow
motion | lubrication | lubrication lubricant oil friction reduce smooth surfaces
motion | friction_texture | rough rougher friction traction grip sandpaper
motion | friction_smooth | smooth smoother ice slippery friction less
motion | elastic | elastic spring stretched compressed potential stored energy
motion | pendulum | pendulum length period swinging potential kinetic gravity
motion | screw_advantage | screw threads inclined plane distance force
motion | gear | gear gears teeth rotation direction wheel
motion | hydraulic | hydraulic pressure fluid liquid piston force
motion | resistance_length | resistance longer thin wire current electricity
motion | conductivity_temperature | metal conductor heat temperature resistance current
motion | electrical_safety | safety electricity wet water shock insulation rubber
motion | fuse | fuse breaker circuit excessive current safety
motion | electromagnet_strength | electromagnet more turns coils current iron stronger
motion | static_charge | electrons transfer static charge friction rubbing balloon
motion | electrical_attraction | opposite unlike charges attract attraction positive negative
motion | electrical_repulsion | like same charges repel repulsion positive negative
motion | oscillation | oscillation vibration back forth wave particles
motion | mechanical_wave | mechanical wave medium particles sound water seismic
motion | electromagnetic_wave | electromagnetic wave vacuum light space radio
motion | sound_medium | sound solid solids liquid liquids faster air gas
motion | wave_frequency | frequency hertz hz cycles second vibration
motion | wave_speed | wave speed frequency wavelength product
motion | concave_mirror | concave mirror focus converge light image
motion | convex_lens | convex lens converge focus light magnify
motion | concave_lens | concave lens diverge spread light rays
motion | flat_mirror | plane flat mirror reflection image reversed
motion | rough_reflection | diffuse reflection rough scatter light surface
motion | white_light | white light colors spectrum prism rainbow
motion | red_light | red light longest visible wavelength
motion | violet_light | violet purple shortest visible wavelength
space | earth_year | earth revolution year 365 days sun orbit
space | earth_day | earth rotation day 24 hours axis night
space | lunar_month | moon phases lunar month 29 days cycle
space | moon_orbit | moon orbit revolution earth 27 days satellite
space | moon_surface | moon craters impacts atmosphere air water rocks
space | earth_protection | atmosphere protects meteoroids radiation oxygen earth
space | elliptical_orbit | ellipse elliptical orbit planets sun focus
space | seasons_opposite | hemispheres northern southern opposite seasons tilt
space | northern_polaris | polaris north northern star axis navigation
space | equator | equator tropical direct sunlight warm latitude
space | poles | poles polar arctic antarctic indirect sunlight cold
space | star_distance | apparent brightness distance closer farther star
space | stellar_fusion | nuclear fusion hydrogen helium stars sun energy
space | gravity_aggregation | gravity dust gas nebula collapse star formation
space | telescope_location | telescope atmosphere mountains high space clearer light
land | continental_drift | continental drift pangaea fossils coastlines continents fit
land | seafloor_spreading | seafloor spreading ridge youngest rocks divergent magma
land | subduction | subduction dense oceanic plate sinking trench mantle
land | earthquake_waves | seismic waves earthquake seismograph epicenter focus
land | sediment_transport | erosion river water wind ice gravity sediments movement
land | delta | delta river mouth deposition slow water sediment
land | meander | meander river curve outer erosion inner deposition
land | dunes | dune dunes wind sand deposition desert
land | landslide | landslide mudslide gravity slope water erosion
land | beach | beach waves sand erosion deposition coast shoreline
land | cave | cave caves limestone groundwater acidic chemical weathering
land | granite | granite igneous intrusive slow cooling large crystals
land | basalt | basalt igneous extrusive fast cooling small crystals lava
land | obsidian | obsidian volcanic glass fast cooling crystals igneous
land | sandstone | sandstone sedimentary sand cementation layers
land | limestone | limestone sedimentary shells calcium carbonate
land | marble | marble metamorphic limestone heat pressure
land | slate | slate metamorphic shale heat pressure
land | hardness | hardness scratch diamond talc mohs mineral
land | streak | streak mineral powder color porcelain
land | luster | luster mineral reflects light shiny dull metallic
land | relative_age | superposition relative age older bottom younger top layers
land | fault_age | fault intrusion cross cutting younger rock layers
land | topographic | topographic contour contours elevation map steep close
land | elevation | elevation altitude height sea level mountains
land | permeability | permeability permeable connected pores water groundwater
land | porosity | porosity pore pores spaces rock soil groundwater
land | impermeable | impermeable clay bedrock barrier groundwater
land | water_treatment | filtration treatment clean drinking water remove contaminants
land | ocean_currents | ocean currents wind temperature density climate heat
land | deep_current | deep ocean current density cold salty sinking
land | salinity | salinity salt ocean evaporation concentration
land | acidification | ocean acidification carbon dioxide dissolved acidic coral
land | lake_effect | lake effect snow moist cold air water evaporation
land | dew | dew condensation droplets cool surface water vapor
land | frost | frost deposition vapor ice freezing cold
land | water_availability | drought little precipitation dry water shortage
land | flood | flood flooding heavy rainfall runoff saturated soil
land | watershed_pollution | fertilizer runoff pollution algae bloom nutrients water
land | eutrophication | eutrophication algae nutrients fertilizer oxygen fish dead
method | accuracy | accuracy accurate true accepted value calibration measure
method | precision | precision precise repeated measurements consistent close
method | bias | bias biased random representative sample fair
method | replication | repeat replication independent repeated reliable experiment
method | field_observation | field observation natural habitat behavior organisms
method | testable_question | testable measurable experiment hypothesis evidence
method | correlation | correlation relationship data variables graph
method | best_model | model representation limitations simplified scale diagram
method | line_graph | line graph continuous time trends changes
method | bar_graph | bar graph compare categories amounts
method | safety_goggles | goggles glasses eyes splashes chemical safety
method | chemical_safety | chemical chemicals taste eat drink unsafe dangerous poison
method | fire_safety | burner flame fire heat tie hair loose clothing safety
method | humane_treatment | animals humane ethical minimize harm care
energy | calorie_unit | calorie calories energy heat food nutrition unit joules
matter | vaporization | vaporization evaporation boiling liquid gas vapor heating
land | paleontology | paleontology paleontologist fossils ancient organisms extinct sedimentary evidence
method | design_criteria | criterion criteria requirements goal success engineering design solution
method | data_analysis | analysis data interpretation patterns results evidence conclusion
method | mathematical_formula | formula formulas equation equations mathematical relationship quantities variables calculation
motion | circle_radius | radius radii center circle circular distance orbit rotation
life | dietary_fiber | fiber fibre cellulose plants food intestines roughage nutrition
method | scientific_theory | theory theories explanation evidence tested supported natural phenomena
method | scientific_law | law laws pattern patterns observations repeated mathematical relationship
'''

# Explicit negative knowledge within an otherwise shared conceptual context.
# Sound requires matter, unlike light; classical closed-system conservation
# excludes creating/destroying the conserved quantity. These members keep the
# same broad science-topic component but reverse the fact-specific component.
NEGATIVE_MEMBERS = {
    ('motion', 'vacuum_sound'): ('vacuum', 'space'),
    ('energy', 'conservation'): ('created', 'destroyed'),
    ('matter', 'conservation_mass'): ('created', 'destroyed'),
}

# These axes encode contrasting scientific categories. They are specified by
# hand, not inferred from text or chosen using question-answer measurements.
CONTRASTS = {}
for axis, positives, negatives in (
    ('renewability', 'renewables solar wind hydroelectric geothermal biomass solar_cell',
     'nonrenewables fossilfuels'),
    ('electrical_conduction', 'metals conductors copper aluminum',
     'nonmetals insulators rubber'),
    ('thermal_conduction', 'conduction heat_conducting', 'insulation heat_insulating'),
    ('melting_freezing', 'melting', 'freezing'),
    ('evaporation_condensation', 'evaporation evaporation_water', 'condensation clouds dew'),
    ('acid_base', 'acid', 'base'),
    ('charge_carrier', 'protons', 'electrons'),
    ('bond_type', 'covalent oxygenbond', 'ionic salt_formula'),
    ('physical_chemical_change', 'physical_change', 'chemical_change rust oxidation'),
    ('energy_form', 'kinetic', 'potential gravitational_potential elastic'),
    ('heat_reaction', 'exothermic', 'endothermic'),
    ('thermal_surface', 'dark_surface', 'light_surface'),
    ('force_balance', 'balanced', 'unbalanced'),
    ('circuit_completeness', 'closed_circuit', 'open_circuit'),
    ('circuit_topology', 'series', 'parallel'),
    ('wave_polarization', 'transverse', 'longitudinal'),
    ('light_transmission', 'transparent', 'opaque'),
    ('light_bending', 'reflection flat_mirror', 'refraction'),
    ('lens_shape', 'convex_lens', 'concave_lens'),
    ('charge_interaction', 'electrical_attraction', 'electrical_repulsion'),
    ('rotation_orbit', 'rotation earth_day', 'revolution earth_year'),
    ('season', 'summer', 'winter'),
    ('sun_direction', 'sunrise', 'sunset'),
    ('air_pressure_regime', 'high_pressure', 'low_pressure'),
    ('air_convection', 'warm_air', 'cold_air'),
    ('plate_boundary', 'divergent seafloor_spreading', 'convergent subduction'),
    ('soil_particles', 'sand', 'clay'),
    ('experimental_role', 'independent', 'dependent'),
    ('experimental_quality', 'accuracy', 'precision'),
    ('reproduction_type', 'sexual meiosis', 'asexual cell_division'),
    ('trait_origin', 'heredity dna_structure', 'acquired'),
    ('trophic_role', 'producers', 'consumers'),
    ('diet', 'herbivores herbivore_teeth', 'carnivores predator_features'),
    ('symbiosis_effect', 'mutualism pollinator_mutualism symbiosis_lichen',
     'parasitism parasite_examples'),
    ('cell_transport', 'diffusion osmosis', 'active_transport'),
    ('friction_surface', 'friction_texture', 'friction_smooth lubrication'),
    ('laboratory_practice', 'safety safety_goggles sanitation', 'unsafe chemical_safety'),
):
    for name in positives.split():
        CONTRASTS[name] = ('contrast:' + axis, 1.0)
    for name in negatives.split():
        CONTRASTS[name] = ('contrast:' + axis, -1.0)


# Words whose unusually absolute or unscientific claims are weak negative evidence.
# These are deliberately small priors, not rules that override science matching.
PRIORS = {
    'only': -0.16, 'always': -0.13, 'never': -0.14, 'all': -0.025,
    'instantly': -0.16, 'immediately': -0.025, 'completely': -0.06,
    'magically': -0.6, 'nothing': -0.1, 'every': -0.045,
    'because': 0.025, 'energy': 0.02, 'evidence': 0.035,
    'measured': 0.025, 'observations': 0.025, 'data': 0.025,
    'same': 0.015, 'different': 0.01,
}
# "Avoid" denotes a hazardous action, not a request to negate all science
# similarity. Its own conceptual neighborhood supplies that positive meaning.
NEGATION = {'not', 'no', 'never', 'non', 'except', 'least', 'incorrect',
            'incorrectly', 'false', 'inaccurate', 'unlikely', 'untrue',
            'unsupported', 'inconsistent'}

# Common variants are explicit equivalences, independent of any observed data.
ALIASES = {
    'co2': 'dioxide', 'h2o': 'water', 'o2': 'oxygen', 'n2': 'nitrogen',
    'dna': 'dna', 'ribonucleic': 'rna', 'deoxyribonucleic': 'dna',
    'photosynthetic': 'photosynthesis', 'photosynthesize': 'photosynthesis',
    'photosynthesizes': 'photosynthesis', 'respire': 'respiration',
    'respiring': 'respiration', 'bacterial': 'bacteria', 'fungal': 'fungi',
    'digest': 'digestion', 'consume': 'consumers', 'react': 'reaction',
    'mix': 'mixture', 'blow': 'wind', 'analyze': 'analysis',
    'goose': 'birds', 'mouse': 'mammals', 'person': 'humans', 'child': 'humans',
    'funguses': 'fungi', 'bacterium': 'bacteria', 'microorganism': 'microorganisms',
    'herbivorous': 'herbivores', 'carnivorous': 'carnivores',
    'omnivorous': 'omnivores', 'predation': 'predator',
    'camouflaged': 'camouflage', 'camouflaging': 'camouflage',
    'genetics': 'genetic', 'genetically': 'genetic', 'hereditary': 'heredity',
    'chromosomal': 'chromosomes', 'reproductive': 'reproduction',
    'unicellular': 'unicellular', 'multicellular': 'tissues',
    'mitochondrial': 'mitochondria', 'nuclei': 'nucleus',
    'artery': 'arteries', 'vein': 'veins', 'capillary': 'capillaries',
    'neurological': 'nervous', 'neuron': 'nerves', 'neurons': 'nerves',
    'hormonal': 'hormones', 'antibiotic': 'bacteria', 'antibiotics': 'bacteria',
    'immunization': 'vaccination', 'vaccines': 'vaccination', 'vaccine': 'vaccination',
    'mammalian': 'mammals', 'reptilian': 'reptiles',
    'amphibious': 'amphibians', 'insectivores': 'insects',
    'vertebrae': 'backbone', 'spinal': 'spinal',
    'evolving': 'evolution', 'evolutionary': 'evolution',
    'adaptive': 'adaptation', 'adapted': 'adaptations',
    'pollinate': 'pollination', 'pollinated': 'pollination',
    'germinating': 'germination', 'germinated': 'germination',
    'decompose': 'decomposition', 'decomposing': 'decomposition',
    'decomposed': 'decay', 'decomposes': 'decomposers',
    'evaporating': 'evaporation', 'evaporated': 'evaporation',
    'condensing': 'condensation', 'condensed': 'condensation',
    'freezes': 'freezing', 'froze': 'freezing', 'frozen': 'ice',
    'melts': 'melting', 'melted': 'melting',
    'sublimates': 'sublimation', 'sublimated': 'sublimation',
    'eroded': 'erosion', 'eroding': 'erosion', 'erodes': 'erosion',
    'weathered': 'weathering', 'weathers': 'weathering',
    'sediments': 'sediment', 'sedimentation': 'deposition',
    'igneous': 'igneous', 'metamorphism': 'metamorphic',
    'tectonics': 'tectonic', 'seismic': 'earthquake',
    'erupt': 'eruption', 'erupted': 'eruption', 'erupting': 'eruption',
    'volcanism': 'volcano', 'volcanic': 'volcano',
    'astronomer': 'telescope', 'astronomers': 'telescope',
    'lunar': 'moon', 'solar': 'solar', 'terrestrial': 'rocky',
    'revolves': 'revolution', 'revolving': 'revolution',
    'rotates': 'rotation', 'rotating': 'rotation',
    'orbits': 'orbit', 'orbiting': 'orbit',
    'gravitational': 'gravity', 'gravitation': 'gravity',
    'magnetized': 'magnet', 'magnetism': 'magnetic',
    'electromagnetic': 'electromagnetic',
    'conductor': 'conductors', 'insulator': 'insulators',
    'conducting': 'conductors', 'insulating': 'insulation',
    'reflecting': 'reflection', 'reflected': 'reflection',
    'refracted': 'refraction', 'refracting': 'refraction',
    'absorbing': 'absorption', 'absorbed': 'absorption',
    'vibrate': 'vibrations', 'vibrating': 'vibrations',
    'vibrates': 'vibrations', 'oscillate': 'oscillation',
    'oscillating': 'oscillation', 'compressions': 'compression',
    'compress': 'compression', 'compressed': 'compressed',
    'accelerate': 'acceleration', 'accelerating': 'acceleration',
    'accelerates': 'acceleration', 'accelerated': 'acceleration',
    'deceleration': 'slow', 'decelerates': 'slow',
    'thermal': 'heat', 'heating': 'heat', 'heated': 'heat',
    'electrical': 'electricity', 'electrically': 'electricity',
    'renewables': 'renewable', 'nonrenewables': 'nonrenewable',
    'sustainable': 'renewable', 'petrol': 'gasoline',
    'hydroelectricity': 'hydroelectric', 'photovoltaic': 'solar',
    'biodegradable': 'decomposition', 'recyclable': 'recycling',
    'recycled': 'recycling', 'reusing': 'reuse',
    'contaminated': 'pollution', 'contaminants': 'pollutants',
    'contamination': 'pollution', 'polluted': 'pollution',
    'dissolving': 'dissolve', 'dissolved': 'dissolve',
    'solutes': 'solute', 'solvents': 'solvent',
    'chemically': 'chemical', 'physically': 'physical',
    'bonded': 'bonds', 'bonding': 'bonds',
    'ions': 'ionic', 'ion': 'ionic', 'ionization': 'ionic',
    'negatively': 'negative', 'positively': 'positive',
    'denser': 'density', 'densest': 'density',
    'floating': 'float', 'floats': 'float',
    'sinking': 'sink', 'sinks': 'sink',
    'experimentally': 'experiment', 'experimental': 'experiment',
    'investigate': 'investigation', 'investigating': 'investigation',
    'research': 'investigation', 'researchers': 'scientists',
    'scientific': 'evidence', 'scientifically': 'evidence',
    'hypotheses': 'hypothesis', 'observing': 'observation',
    'observed': 'observation', 'conclude': 'conclusion',
    'concluded': 'conclusion', 'concluding': 'conclusion',
    'predict': 'prediction', 'predicting': 'prediction',
    'predicts': 'prediction', 'predicted': 'prediction',
    'quantitative': 'measurement', 'qualitative': 'observation',
    'reproducible': 'repeated', 'replicate': 'repeated',
    'replicated': 'repeated', 'replication': 'repeated',
    'centimeter': 'centimeters', 'kilogram': 'kilograms',
    'milliliter': 'milliliters', 'millimetres': 'millimeters',
    'millimetre': 'millimeters', 'millimeter': 'millimeters',
    'kilometres': 'kilometers', 'kilometre': 'kilometers',
    'kilometer': 'kilometers',
    'litres': 'liters', 'metres': 'meters', 'centigrade': 'celsius',
    "doesn't": 'not', "isn't": 'not', "aren't": 'not',
    "wouldn't": 'not', "shouldn't": 'not', "cannot": 'not',
    "can't": 'not', "couldn't": 'not', "don't": 'not',
    "won't": 'not', "didn't": 'not', "hasn't": 'not', "haven't": 'not',
    "hadn't": 'not', "wasn't": 'not', "weren't": 'not', "mustn't": 'not',
    "needn't": 'not',
}

# Function words have zero semantic vectors, even when they occur in a concept
# description. Keeping them out prevents accidental matches on grammar.
STOP = set('''a an the of to in on at by for from and or but with into as than that
this these those it its they them their there which what how when where why who
will would could should can may might do does did is are was were be been being
have has had having he she his her we our you your i me my some any such one two
three four five six seven eight nine ten each following likely most best mainly
usually often generally sometimes primarily example examples shows shown show
statement statements describes described describe explains explain explanation
happens happen happenes occurs occur result results cause causes caused causing
increase increases increased increasing decrease decreases decreased decreasing
more less greater greatest smaller smallest larger largest higher highest lower
lowest new old other another first second third last next before after during
then now over under between through about around because therefore also both
same different much many several given certain student students class school
scientist scientists researcher researchers investigation investigation's
object objects thing things type types kind kinds process processes system
systems part parts use uses used using make makes made making get gets change
changes changed changing help helps helped need needs needed number amount
able ability determine determines determining correct answer question identify
choose true necessary important purpose probably possible available below above
without together per up down back forth left right near far very enough seen
found find occur happened happen happening based related include includes
including contain contains containing called known taking takes take took time
long short large small little big greater better single special specific
'''.split())
# These are not empty grammatical words in science: they distinguish atomic
# number, rates, experimental controls, reproduction, and counts of body parts.
STOP.difference_update({'one', 'two', 'three', 'four', 'five', 'six', 'seven',
                        'eight', 'nine', 'ten', 'number', 'amount', 'time',
                        'same', 'different', 'single'})


def entries():
    for line in GROUPS.splitlines():
        if line.strip():
            topic, name, text = [part.strip() for part in line.split('|')]
            yield topic, name, text.split()


def lexical_seed():
    words = set(PRIORS) | NEGATION | STOP | set(ALIASES) | set(ALIASES.values())
    for _, _, row in entries():
        words.update(row)
    return words


def morphology():
    return aliases_for(lexical_seed(), STOP)


def vocabulary():
    words = lexical_seed()
    forms = morphology()
    words.update(forms)
    words.update(forms.values())
    return sorted(words)
